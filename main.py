import kagglehub

# Download latest version
path = kagglehub.competition_download('autism-prediction')

print("Path to competition files:", path)