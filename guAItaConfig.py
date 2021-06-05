#Setup the folder where the daily folders are generated.
#It can be a local folder or you can link to a share folder in your Raspberry (using Samba)
#Examples: 
# Samba folder: \\\\192.168.1.64/Share
# Local folder: C:/Images
guAIta_main_folder = "C:/Development/meteor_detector/dataset/test_realtime"
guAIta_start_time="2100"
guAIta_end_time = "0800"
guAIta_obs_id = "Pujalt"
guAIta_URL = "https://xxxxxxxxx.execute-api.eu-west-1.amazonaws.com/Prod/uploadmeteor"
guAIta_enable_telegram=True
guAIta_enable_extra_log=True
guAIta_threshold = 0.85
guAIta_sleep_between_imgs = 10
