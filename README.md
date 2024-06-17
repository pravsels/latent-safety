# latent-safety
# latent-safety


# generating image-based pickle files:
conda activate safe-rl
python scripts/generate_data.py

# training models
python scripts.train_classifier.py
python scripts.train_classifier_img.py
python scripts.train_dynamics.py
python scripts.train_dynamics_img.py


# getting grid-based solution
conda activate odp
python scripts/toy.py