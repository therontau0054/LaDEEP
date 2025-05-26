from torch.utils.tensorboard import SummaryWriter

class Tensorboard_Logging(object):
    def __init__(self, log_dir, comment = "", filename_suffix = ""):
        self.writer = SummaryWriter(
                        log_dir = log_dir,
                        comment = comment,
                        filename_suffix = filename_suffix
                    )
    
    def write_2d_figure(self, tag, scalar_value, global_step):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def write_model_graph(self, model, input):
        self.writer.add_graph(model, input)

    def writer_close(self):
        self.writer.close()
