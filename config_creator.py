from translator import Translator


class ConfigCreator:
    @staticmethod
    def create_config_object(config):
        config_object = {"input":[],"output":"som","scenes":0,"name":None,"ag":"row"}
        #ag: row, col, mean
        if isinstance(config["input"], str):
            config_object["input"] = Translator.get_columns_by_input_info(config["input"])
        else:
            config_object["input"] = config["input"]

        for a_prop in ["output","scenes","name","ag"]:
            if a_prop in config:
                config_object[a_prop] = config[a_prop]

        if config_object["name"] is None:
            if isinstance(config["input"], str):
                config_object["name"] = config["input"]
            else:
                config_object["name"] = Translator.get_input_name(config)
            scene_part = config_object['scenes']
            if type(scene_part) == list:
                scene_part = len(config_object['scenes'])

            config_object["name"] = f"{config_object['name']}_{scene_part}"

        return config_object
