def encode(message):
    message = message + "\r\n"
    return message.encode()


def decode(message):
    """ bytes to str; strip carriage return """

    if type(message) is list:
        return [decode(msg) for msg in message]

    return message.decode().strip()


def ismatec_to_flow(pct_rate):
    """ calibration curve from ismatec output fraction to flow in mL/min (0-100%)"""
    mL_per_min_rate = 0.0144 * pct_rate
    return mL_per_min_rate


def flow_to_ismatec(mL_per_min_rate):
    """ calibration curve from flow in mL/min to ismatec output fraction (0-100%)"""
    pct_rate = mL_per_min_rate / 0.0144
    return pct_rate


def proportion_to_flow(rate):
    """ calibration curve from ismatec output proportion (0,1) to flow in mL/min """
    mL_per_min_rate = 1.44 * rate
    return mL_per_min_rate


def flow_to_proportion(mL_per_min_rate):
    """ calibration curve from flow in mL/min to ismatec output proportion (0,1) """
    pct_rate = mL_per_min_rate / 1.44
    return pct_rate
