import json
from stravalib.client import Client
import webbrowser
import os
os.environ["SILENCE_TOKEN_WARNINGS"] = "true"

# Got main logic from:
# https://github.com/stravalib/stravalib/blob/main/docs/get-started/how-to-get-strava-data-python.md
def login():
    if os.path.isfile("secrets/client_secrets.txt"):
        with open("secrets/client_secrets.txt", "r") as f:
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

    if not os.path.exists("secrets/strava_token.json") or "STRAVA_CLIENT_REFRESH_TOKEN" in os.environ:
        request_scope = ["read_all", "profile:read_all", "activity:read_all"]
        redirect_url = "http://127.0.0.1:5000/authorization"
        url = client.authorization_url(
            client_id=client_id,
            redirect_uri=redirect_url,
            scope=request_scope,
        )
        webbrowser.open(url)
        print("""You will see a url that looks like this. """,
            """http://127.0.0.1:5000/authorization?state=&code=12323423423423423423423550&scope=read,activity:read_all,profile:read_all,read_all")""",
            """Copy the values between code= and & in the url that you see in the browser. """)
        code = input("Please enter the code that you received: ")
        token_response = client.exchange_code_for_token(
            client_id=client_id, client_secret=client_secret, code=code)
        with open("secrets/strava_token.json", "w") as f:
            json.dump(token_response, f)
    else:
        if os.path.isfile("secrets/strava_token.json"):
            print("You have already authenticated once before. Refreshing your token now.")
            with open("secrets/strava_token.json") as f:
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
