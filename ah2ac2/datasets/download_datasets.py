import huggingface_hub
from huggingface_hub import snapshot_download


def download_dataset(local_dir="./data"):
    huggingface_hub.login(token="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")
    snapshot_download(repo_id="ah2ac2/datasets", repo_type="dataset", local_dir=local_dir)


if __name__ == '__main__':
    download_dataset()
