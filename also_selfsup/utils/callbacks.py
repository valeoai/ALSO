from pytorch_lightning.callbacks import TQDMProgressBar


class CustomProgressBar(TQDMProgressBar):
    def on_train_epoch_start(self, trainer, *args):
        super().on_train_epoch_start(trainer, *args)
        self.main_progress_bar.reset()

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items