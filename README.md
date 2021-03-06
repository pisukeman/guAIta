![guAIta](https://guaita.s3-eu-west-1.amazonaws.com/logo_small.jpg)

# guAIta
guAIta is an opensource project to detect meteors using deep learning models

guAIta is designed to access to images generated by All Sky software (https://github.com/thomasjacquin/allsky) and automatically infer if the image contains a meteor.

Project structure:
- dataPreparation.py: Set of python functions used to pre-process images
- guAIta.py: Python script to infer over the images in a specific folder
- guAItaConfig.py: Configuration file to guAIta.py scripts
- guAIta_conda_env.yml: Yaml file with all the dependencies needed to run guAIta.py in a local machine
- Training_and_Inference_examples:
  -  jpynb and html files with examples of how the model was trained using fast.ai
- Models:
  -  pkl file with the latest version of the model
 -  AWS_Lambda_functions
	 -  Lambda functions to genereate a backend to notify the meteors detected  
  
  
If you need more information about the project contact me at davidregordosa@gmail.com or @pisukeman on twitter.

