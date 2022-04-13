import hvplot.pandas
import asyncio
import zmq.asyncio
import pandas as pd
from datetime import datetime
from functools import partial

import streamz
import hvplot.streamz
from streamz.dataframe import DataFrame

import panel as pn
import holoviews as hv
from bokeh.plotting import curdoc
from holoviews.streams import Pipe, Buffer

hv.extension("bokeh")

# set up ZMQ subscriber socket
DASHBOARD_PORT = 2345
DASHBOARD_ADDRESS = "127.0.0.1"

context = zmq.asyncio.Context.instance()
socket = context.socket(zmq.SUB)
socket.connect(f"tcp://{DASHBOARD_ADDRESS}:{DASHBOARD_PORT}")
socket.setsockopt(zmq.SUBSCRIBE, b"")

# use a streamz.DataFrame to stream data from zmq socket
source = streamz.Stream()
# example = pd.DataFrame({'pH': [], 'temperature': []})
example = pd.DataFrame({"current": [], "potential": [], "elapsed_time": []})
df = DataFrame(source, example=example)
# df = source.to_dataframe(example=example).window(1000).full()

# pipe = Pipe(data=example)
# source.sliding_window(2).map(pd.concat).sink(pipe.send) # Connect streamz to the Pipe


async def loop():
    """ stream pandas dataframe chunks from zmq socket... """
    while True:
        new_data = await socket.recv_pyobj()
        source.emit(new_data)


# run the zmq client loop in the background
asyncio.create_task(loop())

# set up pyviz.panel layout
options = dict(width=800, height=300, show_grid=True, xlabel="elapsed time (s)")
pn.Column(
    # pn.panel(hv.DynamicMap(hv.Curve, streams=[Buffer(df.pH)]).opts(title='pH', **options)),
    # pn.panel(hv.DynamicMap(hv.Curve, streams=[Buffer(df.temperature)]).opts(title='temperature', **options))
    pn.panel(
        hv.DynamicMap(
            partial(hv.Curve, kdims=["elapsed_time"], vdims=["potential"]),
            streams=[Buffer(df, index=False, length=4000)],
        ).opts(title="pH", **options)
    ),
    pn.panel(
        hv.DynamicMap(
            partial(hv.Curve, kdims=["potential"], vdims=["current"]),
            streams=[Buffer(df, index=False, length=4000)],
        ).opts(title="temperature", **options)
    )
    # pn.panel(df.hvplot(x='potential', y='current', backlog=1000).opts(title='echem', **options)),
    # pn.panel(hv.DynamicMap(hv.Curve, streams=[pipe]).opts(title='temperature', **options))
).servable()
