import os
import re
import slack
import time
import asyncio
import concurrent
from datetime import datetime

# check if there is a proxy between localhost and the internet
PROXY = os.environ.get("https_proxy")

BOT_TOKEN = open("slacktoken.txt", "r").read().strip()

bot_patterns = {"sdc": "<@UHT11TM6F>", "ctl": "<@UHNHM7198>"}


class CommandRegistry(set):
    """ wrap a set with a decorator to register commands """

    def register(self, method):
        self.add(method.__name__)
        return method

    def __call__(self, method):
        """ overload __call__ instead of having an explicit register method """
        return self.register(method)


class SlackBot(object):
    command = CommandRegistry()

    def __init__(self, name="sdc", token=None):
        self.name = name
        # self._pattern = f'@{self.name}'
        self._pattern = bot_patterns.get(name)  # '<@UHT11TM6F>'

        if token is None:
            self.token = BOT_TOKEN
        else:
            self.token = token

        slack.RTMClient.run_on(event="message")(self.handle_message)

    async def handle_message(self, **payload):
        print(payload)
        data = payload["data"]
        text = data.get("text", "")
        print(text)

        bot_reference = re.match(self._pattern, text)
        if bot_reference:
            print("reference to bot!")

            # handle bot commands
            _, end = bot_reference.span()
            m = text[end:]

            try:
                command, args = m.strip().split(None, 1)
            except ValueError:
                command, args = m.strip(), None

            print(command)

            if command not in self.command:
                user = data.get("user")
                r = f"sorry <@{user}>, I didn't understand what you meant by `{m.strip()}`..."
                print(r)
                web_client = payload["web_client"]

                web_client.chat_postMessage(
                    channel=data["channel_id"], text=r, thread_ts=data["ts"]
                )

            else:
                # msgdata = {'username': data['user'], 'channel': data['channel_id']}
                # self.log(text)

                # dispatch command method by name
                # await getattr(self, command)(ws, msgdata, args)
                web_client = payload["web_client"]
                await getattr(self, command)(args, data, web_client)

    @command
    async def echo(self, text, data, web_client):
        r = f"recieved command {text}"
        channel = data.get("channel_id")
        if channel is None:
            channel = data.get("channel")
        print(r)
        web_client.chat_postMessage(channel=channel, text=r, thread_ts=data["ts"])

    async def main(self):

        if PROXY is not None:
            proxy_address = f"http://{PROXY}"
        else:
            proxy_address = None

        self.loop = asyncio.get_event_loop()
        self.client = slack.RTMClient(
            token=self.token, proxy=proxy_address, run_async=True, loop=self.loop
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        await asyncio.ensure_future(self.client.start())


if __name__ == "__main__":
    bot = SlackBot()
    asyncio.run(bot.main())
