#############################################################
########   please use official docker conainer     ##########
#############################################################

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

MODEL_ID = 'damo/mplug_image-captioning_coco_base_en'

def mPLUG_captioning(image_list: list) -> list:
    pipeline_caption = pipeline(Tasks.image_captioning, model=MODEL_ID, config_file="customized_config.yaml")  # specify the setting with customized config yaml
    result = pipeline_caption(image_list)
    values_list = [item['caption'] for item in result]
    
    return values_list