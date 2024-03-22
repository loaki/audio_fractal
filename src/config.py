import configparser
import soundcard as sc
import argparse


from record import record_sound


class Conf:
    def config_speaker(self, config: configparser.ConfigParser):
        print("\n###\nChoose default speaker")
        speakers = sc.all_speakers()
        for i, speaker in enumerate(speakers):
            print(i, speaker)
        index = input()
        try:
            config["DEFAULT"]["speaker"] = speakers[int(index)].name
            with open("config.ini", "w") as configfile:
                config.write(configfile)
        except Exception as e:
            print(e)
            self.config_speaker(config)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    conf = Conf()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cs", "--config_speaker", action="store_true", help="Configure default speaker"
    )
    args = parser.parse_args()
    for arg in list(filter(lambda a: a[1], args._get_kwargs())):
        if hasattr(Conf, arg[0]):
            class_method = getattr(conf, arg[0])
            class_method(config)
