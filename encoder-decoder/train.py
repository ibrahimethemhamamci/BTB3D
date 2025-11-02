from omegaconf import OmegaConf
from src.trainer import VideoTokenizerTrainer
from argparse import ArgumentParser

def get_config_cli():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf

def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--trainer_config_path", type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    trainer = VideoTokenizerTrainer(model_config=get_config(args.model_config_path), trainer_config=get_config(args.trainer_config_path))
    trainer.train()
    

if __name__ == "__main__":
    main()
