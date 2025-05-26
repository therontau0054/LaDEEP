import os
import logging
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from PIL import Image
from tqdm import tqdm

from core.data_loader import LaDEEPDataLoader
from core.ladeep import LaDEEP
from core.losses import loss_p, loss_r

from utils.tensorboard_logging import Tensorboard_Logging
from utils.diff_recover import diff_recover


def warm_up(
        net,
        train_datas,
        coordinate_weights,
        warm_up_steps,
        init_lr,
        max_lr,
        device
):
    params_p_unloading = {
        n: p for n, p in net.named_parameters()
        if "unloading" in n
    }
    params_r = {
        n: p for n, p in net.named_parameters()
        if p.requires_grad
        and ("cse" in n or "csd" in n)
    }

    params_p_loading = {
        n: p for n, p in net.named_parameters()
        if p.requires_grad
        and n not in params_p_unloading.keys()
        and n not in params_r.keys()
    }

    optimizer_p_loading = optim.Adam(
        params_p_loading.values(),
        lr = init_lr
    )

    optimizer_p_unloading = optim.Adam(
        params_p_unloading.values(),
        lr = init_lr
    )
    optimizer_r = optim.Adam(
        params_r.values(),
        lr = init_lr
    )

    lr_step_gamma = (max_lr / init_lr) ** (1 / warm_up_steps)

    scheduler_p_loading = StepLR(
        optimizer_p_loading,
        step_size = 1,
        gamma = lr_step_gamma
    )

    scheduler_p_unloading = StepLR(
        optimizer_p_unloading,
        step_size = 1,
        gamma = lr_step_gamma
    )

    scheduler_r = StepLR(
        optimizer_r,
        step_size = 1,
        gamma = lr_step_gamma
    )

    for _ in tqdm(range(warm_up_steps)):
        net = net.train()
        for i, train_data in enumerate(train_datas):
            strip, mould, section, params, loaded_strip, unloaded_strip = \
                list(
                    map(
                        lambda x: x.to(device), train_data
                    )
                )
            recover_section, pred_loaded_strip, pred_unloaded_strip = net(strip, mould, section, params)
            train_loss_p_loading = loss_p(loaded_strip, pred_loaded_strip, coordinate_weights[0])
            train_loss_p_unloading = loss_p(unloaded_strip, pred_unloaded_strip, coordinate_weights[1])
            train_loss_r = loss_r(section, recover_section)

            optimizer_p_loading.zero_grad()
            optimizer_p_unloading.zero_grad()
            optimizer_r.zero_grad()
            train_loss_p_loading.backward(retain_graph = True)
            train_loss_p_unloading.backward(retain_graph = True)
            train_loss_r.backward()
            optimizer_p_loading.step()
            optimizer_p_unloading.step()
            optimizer_r.step()
        scheduler_p_loading.step()
        scheduler_p_unloading.step()
        scheduler_r.step()


def train(
        train_dataloader,
        eval_dataloader,
        device,
        exp_name,
        tl_writer
):
    train_datas = DataLoader(
        train_dataloader,
        batch_size = train_batch_size,
        shuffle = True,
        num_workers = num_workers
    )

    eval_datas = DataLoader(
        eval_dataloader,
        batch_size = eval_batch_size,
        num_workers = num_workers
    )

    coordinate_weights = [
        [
            train_dataloader.scale_factor_for_load[i] - train_dataloader.scale_factor_for_load[i + 3]
            for i in range(3)
        ],
        [
            train_dataloader.scale_factor_for_unload[i] - train_dataloader.scale_factor_for_unload[i + 3]
            for i in range(3)
        ]
    ]

    checkpoint_save_path = os.path.join(checkpoints_path, exp_name)
    os.makedirs(checkpoint_save_path, exist_ok = True)

    torch.cuda.manual_seed(seed)

    net = LaDEEP().to(device)

    if warm_up_flag:
        warm_up(
            net,
            train_datas,
            coordinate_weights,
            int(epochs * warm_up_step_ratio),
            warm_up_init_lr,
            warm_up_max_lr,
            device
        )

    params_p_unloading = {
        n: p for n, p in net.named_parameters()
        if "unloading" in n
    }
    params_r = {
        n: p for n, p in net.named_parameters()
        if p.requires_grad
        and ("cse" in n or "csd" in n)
    }

    params_p_loading = {
        n: p for n, p in net.named_parameters()
        if p.requires_grad
        and n not in params_p_unloading.keys()
        and n not in params_r.keys()
    }

    optimizer_p_loading = optim.Adam(
        params_p_loading.values(),
        lr = lr_p_loading,
        weight_decay = weight_decay_p_loading
    )

    optimizer_p_unloading = optim.Adam(
        params_p_unloading.values(),
        lr = lr_p_unloading,
        weight_decay = weight_decay_p_unloading
    )

    optimizer_r = optim.Adam(
        params_r.values(),
        lr = lr_r,
        weight_decay = weight_decay_r
    )

    scheduler_p_loading = CosineAnnealingLR(
        optimizer_p_loading,
        epochs,
        eta_min = lr_p_loading / lr_p_loading_decay_min
    )

    scheduler_p_unloading = CosineAnnealingLR(
        optimizer_p_unloading,
        epochs,
        eta_min = lr_p_unloading / lr_p_unloading_decay_min
    )

    scheduler_r = CosineAnnealingLR(
        optimizer_r,
        epochs,
        eta_min = lr_r / lr_r_decay_min
    )

    len_train_datas = len(train_datas)
    len_eval_datas = len(eval_datas)

    min_train_loss_p_loading = 1 << 30
    min_train_loss_p_unloading = 1 << 30
    min_train_loss_r = 1 << 30
    min_eval_loss_p_loading = 1 << 30
    min_eval_loss_p_unloading = 1 << 30
    min_eval_loss_r = 1 << 30

    for epoch in tqdm(range(epochs)):
        net = net.train()

        mean_train_loss_p_loading = 0.
        mean_train_loss_p_unloading = 0.
        mean_train_loss_r = 0.
        for i, train_data in enumerate(train_datas):
            strip, mould, section, params, loaded_strip, unloaded_strip = \
                list(
                    map(
                        lambda x: x.to(device), train_data
                    )
                )
            recover_section, pred_loaded_strip, pred_unloaded_strip = net(strip, mould, section, params)
            train_loss_p_loading = loss_p(loaded_strip, pred_loaded_strip, coordinate_weights[0])
            train_loss_p_unloading = loss_p(unloaded_strip, pred_unloaded_strip, coordinate_weights[1])
            train_loss_r = loss_r(section, recover_section)

            mean_train_loss_p_loading += train_loss_p_loading.data
            mean_train_loss_p_unloading += train_loss_p_unloading.data
            mean_train_loss_r += train_loss_r.data

            tl_writer.write_2d_figure(
                "train/train_loss_p_loading",
                train_loss_p_loading.data,
                epoch * len_train_datas + i
            )
            tl_writer.write_2d_figure(
                "train/train_loss_p_unloading",
                train_loss_p_unloading.data,
                epoch * len_train_datas + i
            )
            tl_writer.write_2d_figure(
                "train/train_loss_r",
                train_loss_r.data,
                epoch * len_train_datas + i
            )

            min_train_loss_p_loading = min(train_loss_p_loading.data, min_train_loss_p_loading)
            min_train_loss_p_unloading = min(train_loss_p_unloading.data, min_train_loss_p_unloading)
            min_train_loss_r = min(train_loss_r.data, min_train_loss_r)

            tl_writer.write_2d_figure(
                "train/min_train_loss_p_loading",
                min_train_loss_p_loading.data,
                epoch * len_train_datas + i
            )
            tl_writer.write_2d_figure(
                "train/min_train_loss_p_unloading",
                min_train_loss_p_unloading.data,
                epoch * len_train_datas + i
            )
            tl_writer.write_2d_figure(
                "train/min_train_loss_r",
                min_train_loss_r.data,
                epoch * len_train_datas + i
            )

            optimizer_r.zero_grad()
            optimizer_p_loading.zero_grad()
            optimizer_p_unloading.zero_grad()
            train_loss_p_loading.backward(retain_graph = True)
            train_loss_p_unloading.backward(retain_graph = True)
            train_loss_r.backward()
            optimizer_p_loading.step()
            optimizer_p_unloading.step()
            optimizer_r.step()

        scheduler_p_loading.step()
        scheduler_p_unloading.step()
        scheduler_r.step()

        mean_train_loss_p_loading /= len_train_datas
        mean_train_loss_p_unloading /= len_train_datas
        mean_train_loss_r /= len_train_datas

        tl_writer.write_2d_figure(
            "train/mean_train_loss_p_loading",
            mean_train_loss_p_loading,
            epoch
        )
        tl_writer.write_2d_figure(
            "train/mean_train_loss_p_unloading",
            mean_train_loss_p_unloading,
            epoch
        )
        tl_writer.write_2d_figure(
            "train/mean_train_loss_r",
            mean_train_loss_r,
            epoch
        )

        net = net.eval()
        with torch.no_grad():
            mean_eval_loss_p_loading = 0.
            mean_eval_loss_p_unloading = 0.
            mean_eval_loss_r = 0.
            for i, eval_data in enumerate(eval_datas):
                strip, mould, section, params, loaded_strip, unloaded_strip = \
                    list(
                        map(
                            lambda x: x.to(device), eval_data
                        )
                    )
                recover_section, pred_loaded_strip, pred_unloaded_strip = net(strip, mould, section, params)
                eval_loss_p_loading = loss_p(loaded_strip, pred_loaded_strip, coordinate_weights[0])
                eval_loss_p_unloading = loss_p(unloaded_strip, pred_unloaded_strip, coordinate_weights[1])
                eval_loss_r = loss_r(section, recover_section)

                tl_writer.write_2d_figure(
                    "eval/eval_loss_p_loading",
                    eval_loss_p_loading,
                    epoch * len_eval_datas + i
                )
                tl_writer.write_2d_figure(
                    "eval/eval_loss_p_unloading",
                    eval_loss_p_unloading,
                    epoch * len_eval_datas + i
                )
                tl_writer.write_2d_figure(
                    "eval/eval_loss_r",
                    eval_loss_r,
                    epoch * len_eval_datas + i
                )

                mean_eval_loss_p_loading += eval_loss_p_loading.data
                mean_eval_loss_p_unloading += eval_loss_p_unloading.data
                mean_eval_loss_r += eval_loss_r.data
            mean_eval_loss_p_loading /= len_eval_datas
            mean_eval_loss_p_unloading /= len_eval_datas
            mean_eval_loss_r /= len_eval_datas

            tl_writer.write_2d_figure(
                "eval/mean_eval_loss_p_loading",
                mean_eval_loss_p_loading,
                epoch
            )
            tl_writer.write_2d_figure(
                "eval/mean_eval_loss_p_unloading",
                mean_eval_loss_p_unloading,
                epoch
            )
            tl_writer.write_2d_figure(
                "eval/mean_eval_loss_r",
                mean_eval_loss_r,
                epoch
            )
            if mean_eval_loss_p_loading < min_eval_loss_p_loading:
                min_eval_loss_p_loading = mean_eval_loss_p_loading

            if mean_eval_loss_p_unloading < min_eval_loss_p_unloading:
                saved_params = {
                    n: p for n, p in net.state_dict().items()
                }
                torch.save(
                    saved_params,
                    os.path.join(checkpoint_save_path, "best_model.pth")
                )
                min_eval_loss_p_unloading = mean_eval_loss_p_unloading
                logging.info(f"checkpoint saved at {epoch} with loss {min_eval_loss_p_unloading}")

            if mean_eval_loss_r < min_eval_loss_r:
                min_eval_loss_r = mean_eval_loss_r

    logging.info("training finished")
    logging.info(f"min_eval_loss_p_loading : {min_eval_loss_p_loading}")
    logging.info(f"min_eval_loss_p_unloading : {min_eval_loss_p_unloading}")
    logging.info(f"min_eval_loss_r : {min_eval_loss_r}")


def test(data_path, test_dataloader, device, exp_name):
    coordinate_weights = [
        [
            test_dataloader.scale_factor_for_load[i] - test_dataloader.scale_factor_for_load[i + 3]
            for i in range(3)
        ],
        [
            test_dataloader.scale_factor_for_unload[i] - test_dataloader.scale_factor_for_unload[i + 3]
            for i in range(3)
        ]
    ]

    test_datas = DataLoader(
        test_dataloader,
        batch_size = test_batch_size,
        num_workers = num_workers
    )

    data_save_path = os.path.join(data_path, f"pred_results/{exp_name}")

    pred_loaded_strip_path = os.path.join(
        data_save_path,
        "pred_loaded_strip_line"
    )
    pred_unloaded_strip_path = os.path.join(
        data_save_path,
        "pred_unloaded_strip_line"
    )
    recover_section_path = os.path.join(
        data_save_path,
        "recover_section"
    )

    os.makedirs(pred_loaded_strip_path, exist_ok = True)
    os.makedirs(pred_unloaded_strip_path, exist_ok = True)
    os.makedirs(recover_section_path, exist_ok = True)

    net = LaDEEP().to(device)
    checkpoint_load_path = exp_name.replace("test", "train")
    checkpoint_load_path = os.path.join(checkpoints_path, checkpoint_load_path, "best_model.pth")

    parameters = torch.load(checkpoint_load_path, map_location = device, weights_only = True)

    # net_parameters = net.state_dict()
    # print(f"The count of model parameters: {len(net_parameters.keys())}")
    # print(f"The count of loaded parameters: {len(parameters.keys())}")

    for key, _ in net.named_parameters():
        if key not in parameters.keys():
            print(f"Error checkpoint file, missing parameter: {key}")

    net.load_state_dict(parameters)

    len_test_datas = len(test_datas)

    pred_loaded_strips, pred_unloaded_strips, recover_sections = [], [], []
    net = net.eval()
    with torch.no_grad():
        mean_test_loss_p_loading = 0.
        mean_test_loss_p_unloading = 0.
        mean_test_loss_r = 0.

        for i, test_data in enumerate(test_datas):
            strip, mould, section, params, loaded_strip, unloaded_strip = \
                list(
                    map(
                        lambda x: x.to(device), test_data
                    )
                )
            recover_section, pred_loaded_strip, pred_unloaded_strip = net(strip, mould, section, params)
            test_loss_p_loading = loss_p(loaded_strip, pred_loaded_strip, coordinate_weights[0])
            test_loss_p_unloading = loss_p(unloaded_strip, pred_unloaded_strip, coordinate_weights[1])
            test_loss_r = loss_r(section, recover_section)

            mean_test_loss_p_loading += test_loss_p_loading.data
            mean_test_loss_p_unloading += test_loss_p_unloading.data
            mean_test_loss_r += test_loss_r.data

            pred_loaded_strips.append(pred_loaded_strip.cpu().detach().numpy())
            pred_unloaded_strips.append(pred_unloaded_strip.cpu().detach().numpy())
            recover_sections.append(recover_section.cpu().detach().numpy())

        mean_test_loss_p_loading /= len_test_datas
        mean_test_loss_p_unloading /= len_test_datas
        mean_test_loss_r /= len_test_datas
        logging.info("Test Loss")
        logging.info(f"mean loss for pred loaded strip: {mean_test_loss_p_loading}")
        logging.info(f"mean loss for pred unloaded strip: {mean_test_loss_p_unloading}")
        logging.info(f"mean loss for recover cross section: {mean_test_loss_r}")
        logging.info("")

    pred_loaded_strips = np.concatenate(pred_loaded_strips, axis = 0)
    pred_unloaded_strips = np.concatenate(pred_unloaded_strips, axis = 0)
    recover_sections = np.concatenate(recover_sections, axis = 0)

    for i in range(3):
        pred_loaded_strips[:, i, :] = pred_loaded_strips[:, i, :] * coordinate_weights[0][i] \
                                      + test_dataloader.scale_factor_for_load[i + 3]
        pred_unloaded_strips[:, i, :] = pred_unloaded_strips[:, i, :] * coordinate_weights[1][i] \
                                        + test_dataloader.scale_factor_for_unload[i + 3]

    def transform_and_save_strip(pred_strip, save_path, pred_type = "loading"):
        mean_test_dist_last_point, max_test_dist_last_point, min_test_dist_last_point = 0, 0, 1 << 30
        mean_sample_mean_test_dist, max_sample_mean_test_dist, min_sample_mean_test_dist = 0, 0, 1 << 30

        gt_path = test_dataloader.load_line_paths
        if pred_type == "unloading":
            gt_path = test_dataloader.unload_line_paths

        for j in range(pred_strip.shape[0]):
            pred_points = diff_recover(pred_strip[j, :, :].T)
            with open(gt_path[j]) as f:
                gt_points = np.array(
                    list(
                        map(
                            lambda x: list(
                                map(
                                    lambda y: float(y),
                                    x.split()
                                )
                            ),
                            f.readlines()
                        )
                    )
                )
            distance = np.sqrt(np.sum((pred_points - gt_points) ** 2, axis = 1))
            test_dist_last_point = np.sqrt(np.sum((pred_points[-1] - gt_points[-1]) ** 2))
            mean_test_dist_last_point += test_dist_last_point
            max_test_dist_last_point = max(max_test_dist_last_point, test_dist_last_point)
            min_test_dist_last_point = min(min_test_dist_last_point, test_dist_last_point)

            sample_mean_test_dist = np.mean(distance)
            max_sample_mean_test_dist = max(max_sample_mean_test_dist, sample_mean_test_dist)
            min_sample_mean_test_dist = min(min_sample_mean_test_dist, sample_mean_test_dist)

            mean_sample_mean_test_dist += sample_mean_test_dist

            type_idx = test_dataloader.mould_line_paths[j].find("type")
            type_id = test_dataloader.mould_line_paths[j][type_idx: type_idx + 6]
            type_path = os.path.join(save_path, type_id)
            if not os.path.exists(type_path):
                os.mkdir(type_path)
            file_path = os.path.join(
                str(type_path),
                str(test_dataloader.mould_line_paths[j][-8:])
            )
            with open(file_path, 'w', encoding = "utf8") as w:
                for k in range(pred_points.shape[0]):
                    w.write(f"{pred_points[k][0]} {pred_points[k][1]} {pred_points[k][2]}\n")
        mean_sample_mean_test_dist /= pred_strip.shape[0]
        mean_test_dist_last_point /= pred_strip.shape[0]

        logging.info(f"Test Error for Stage: {pred_type}")
        logging.info(f"mean distance of last point: {mean_test_dist_last_point}")
        logging.info(f"max distance of last point: {max_test_dist_last_point}")
        logging.info(f"min distance of last point: {min_test_dist_last_point}")
        logging.info(f"mean of distance for each sample: {mean_sample_mean_test_dist}")
        logging.info(f"max of distance for each sample: {max_sample_mean_test_dist}")
        logging.info(f"min of distance for each sample: {min_sample_mean_test_dist}")
        logging.info("")

    def transform_and_save_section(pred_section, save_path):
        for j in range(pred_section.shape[0]):
            type_idx = test_dataloader.mould_line_paths[j].find("type")
            type_id = test_dataloader.mould_line_paths[j][type_idx: type_idx + 6]
            type_path = os.path.join(save_path, type_id)
            os.makedirs(type_path, exist_ok = True)
            file_path = os.path.join(
                str(type_path),
                f"{test_dataloader.mould_line_paths[j][-8: -3]}.jpg"
            )
            sec = recover_sections[j, :, :, :] * 255
            sec = np.transpose(sec, (1, 2, 0)).astype(np.uint8).squeeze()
            sec = Image.fromarray(sec)
            sec.save(file_path)

    transform_and_save_strip(pred_loaded_strips, pred_loaded_strip_path, "loading")
    transform_and_save_strip(pred_unloaded_strips, pred_unloaded_strip_path, "unloading")
    transform_and_save_section(recover_sections, recover_section_path)


def main():
    exp_name = f"{mode}_{mode_id}"
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    logs_save_path = os.path.join(logs_path, exp_name)
    if os.path.exists(logs_save_path):
        exit("logs path exists, please set a new dir!")
    os.makedirs(logs_save_path, exist_ok = True)

    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt = '%m-%d %H:%M',
        filename = os.path.join(logs_save_path, "log.log"),
        filemode = 'w'
    )

    strip_path = os.path.join(data_path, "original_strip_line")
    mould_path = os.path.join(data_path, "mould_line")
    section_path = os.path.join(data_path, "strip_section_tiff")
    params_path = os.path.join(data_path, "stretch_bending_params")
    load_path = os.path.join(data_path, "load_strip_line")
    unload_path = os.path.join(data_path, "unload_strip_line")

    if mode == "train":
        train_dataloader = LaDEEPDataLoader(
            strip_path = strip_path,
            mould_path = mould_path,
            section_path = section_path,
            params_path = params_path,
            load_path = load_path,
            unload_path = unload_path,
            mode = "train"
        )
        eval_dataloader = LaDEEPDataLoader(
            strip_path = strip_path,
            mould_path = mould_path,
            section_path = section_path,
            params_path = params_path,
            load_path = load_path,
            unload_path = unload_path,
            mode = "eval"
        )
        tl_writer = Tensorboard_Logging(logs_save_path)
        train(train_dataloader, eval_dataloader, device, exp_name, tl_writer)
        tl_writer.writer_close()

    elif mode == "test":
        test_dataloader = LaDEEPDataLoader(
            strip_path = strip_path,
            mould_path = mould_path,
            section_path = section_path,
            params_path = params_path,
            load_path = load_path,
            unload_path = unload_path,
            mode = "test"
        )

        test(data_path, test_dataloader, device, exp_name)


if __name__ == "__main__":
    from configparser import ConfigParser

    config = ConfigParser()
    config.read("./config.ini", encoding = "utf-8")
    mode = config.get("settings", "mode")
    mode_id = config.getint("settings", "mode_id")
    device_id = config.getint("settings", "device_id")
    data_path = config.get("settings", "data_path")
    checkpoints_path = config.get("settings", "checkpoints_path")
    logs_path = config.get("settings", "logs_path")
    num_workers = config.getint("settings", "num_workers")

    train_batch_size = config.getint("hyper_parameters", "train_batch_size")
    eval_batch_size = config.getint("hyper_parameters", "eval_batch_size")
    test_batch_size = config.getint("hyper_parameters", "test_batch_size")
    epochs = config.getint("hyper_parameters", "epochs")
    lr_p_loading = config.getfloat("hyper_parameters", "lr_p_loading")
    lr_p_unloading = config.getfloat("hyper_parameters", "lr_p_unloading")
    lr_r = config.getfloat("hyper_parameters", "lr_r")
    weight_decay_p_loading = config.getfloat("hyper_parameters", "weight_decay_p_loading")
    weight_decay_p_unloading = config.getfloat("hyper_parameters", "weight_decay_p_unloading")
    weight_decay_r = config.getfloat("hyper_parameters", "weight_decay_r")
    seed = config.getint("hyper_parameters", "seed")
    lr_p_loading_decay_min = config.getfloat("hyper_parameters", "lr_p_loading_decay_min")
    lr_p_unloading_decay_min = config.getfloat("hyper_parameters", "lr_p_unloading_decay_min")
    lr_r_decay_min = config.getfloat("hyper_parameters", "lr_r_decay_min")
    warm_up_flag = config.getboolean("hyper_parameters", "warm_up_flag")
    warm_up_step_ratio = config.getfloat("hyper_parameters", "warm_up_step_ratio")
    warm_up_init_lr = config.getfloat("hyper_parameters", "warm_up_init_lr")
    warm_up_max_lr = config.getfloat("hyper_parameters", "warm_up_max_lr")

    main()
