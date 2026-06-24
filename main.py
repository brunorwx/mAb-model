from uvicorn import run


def main() -> None:
    run("src.api.app:app", host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
