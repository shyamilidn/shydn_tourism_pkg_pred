from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("shydnTPkg"))
api.upload_folder(
    folder_path="tourism_projectpp/deployment",     # the local folder containing your files
    repo_id="shyam92/TPP",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
