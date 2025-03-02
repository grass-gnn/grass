from typing import Final, Optional, Dict, TypedDict

from grass.model import GRASSModelConfigDict
from grass.trainer import GRASSTrainerConfigDict


class GRASSConfig:
    def __init__(
        self,
        model_config: GRASSModelConfigDict,
        trainer_config: GRASSTrainerConfigDict,
        task_specific_config: Optional[TypedDict] = None,
        verbose: bool = True,
    ) -> None:
        self.model_config: Final = model_config
        self.trainer_config: Final = trainer_config
        self.task_specific_config: Final = task_specific_config
        if verbose:
            print(self)

    def __str__(self) -> str:
        def to_str(dict: Dict) -> str:
            dict_as_str = ""
            for key in dict:
                value = dict[key]
                dict_as_str += f"{str(key)}: {str(value)}\n"

            return dict_as_str

        config_as_str = (
            f"Model Config\n{to_str(self.model_config)}\n"
            f"Trainer Config\n{to_str(self.trainer_config)}"
        )

        if self.task_specific_config is not None:
            config_as_str += (
                f"\nTask-Specific Config\n{to_str(self.task_specific_config)}"
            )

        return config_as_str
