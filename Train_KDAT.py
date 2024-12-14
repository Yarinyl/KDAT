from Config import Configs
import lightning.pytorch as pl
from Defender.TwoStageDefender import TwoStageDefender
from Defender.TransformerBasesDefender import TransformerBasedDefender

if __name__ == '__main__':
    seed = 42
    pl.seed_everything(seed)

    print("Start TwoStageDefender")
    config = Configs()
    config.model_name = 'Faster'
    faster_defender = TwoStageDefender(config.device, config)
    faster_defender.trainer()

    print("Start TransformerBasedDefender")
    config.model_name = 'DETR'
    detr_defender = TransformerBasedDefender(config.device, config)
    detr_defender.trainer()
