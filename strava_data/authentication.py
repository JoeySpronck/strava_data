import json
from stravalib.client import Client
import webbrowser
import os
os.environ["SILENCE_TOKEN_WARNINGS"] = "true"

# Got main logic from:
# https://github.com/stravalib/stravalib/blob/main/docs/get-started/how-to-get-strava-data-python.md
def login(secrets_folder: str = "secrets") -> Client:
    secrets_path = os.path.join(secrets_folder, "client_secrets.txt")
    if os.path.isfile(secrets_path):
        with open(secrets_path, "r") as f:
            # This file should contain your client_id and client_secret, separated by a comma
            client_id, client_secret = f.read().strip().split(",")
    elif "STRAVA_CLIENT_ID" in os.environ and "STRAVA_CLIENT_SECRET" in os.environ:
        client_id = os.environ["STRAVA_CLIENT_ID"]
        client_secret = os.environ["STRAVA_CLIENT_SECRET"]
    else:
        raise Exception("No client_secrets.txt file found in secrets/ folder, ",
            "and no STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET environment variables set. ",
            "Please create the file or set the environment variables.")
    client = Client()

    if not os.path.exists(os.path.join(secrets_folder, "strava_token.json")) and "STRAVA_CLIENT_REFRESH_TOKEN" not in os.environ:
        request_scope = ["read_all", "profile:read_all", "activity:read_all"]
        redirect_url = "http://127.0.0.1:5000/authorization"
        url = client.authorization_url(
            client_id=client_id,
            redirect_uri=redirect_url,
            scope=request_scope,
        )
        webbrowser.open(url)
        print("After login and authorization, you'll be reirected to a new screen that may show 'Access to ... was denied'.\nThat is expected.\n")
        print("""This page has an url that looks like this. """,
            """http://127.0.0.1:5000/authorization?state=&code=12323423423423423423423550&scope=read,activity:read_all,profile:read_all,read_all")""",
            """Copy the values between 'code=' and '&' in the url that you see in the browser, and return it in the input prompt.""")
        code = input("Please enter the code that you received: ")
        token_response = client.exchange_code_for_token(
            client_id=client_id, client_secret=client_secret, code=code)
        with open(os.path.join(secrets_folder, "strava_token.json"), "w") as f:
            json.dump(token_response, f)
    else:
        if os.path.isfile(os.path.join(secrets_folder, "strava_token.json")):
            print("You have already authenticated once before. Refreshing your token now.")
            with open(os.path.join(secrets_folder, "strava_token.json")) as f:
                token_response = json.load(f)
            refresh_token = token_response["refresh_token"]
        else:
            if "STRAVA_CLIENT_REFRESH_TOKEN" in os.environ:
                refresh_token = os.environ["STRAVA_CLIENT_REFRESH_TOKEN"]
                print("Using STRAVA_CLIENT_REFRESH_TOKEN from environment variable.")
        refresh_response = client.refresh_access_token(
            client_id=client_id,  # Stored in the secrets.txt file above
            client_secret=client_secret,
            refresh_token=refresh_token,  # Stored in your JSON file or env variable
        )
    # Check that the refresh worked
    athlete = client.get_athlete()
    print(f"Hi {athlete.firstname}, authentication successful!")
    return client
