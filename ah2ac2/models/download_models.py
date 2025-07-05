from huggingface_hub import snapshot_download


def download_weights(local_dir="."):
    snapshot_download(repo_id="ah2ac2/baselines", local_dir=local_dir)


if __name__ == '__main__':
    download_weights()
