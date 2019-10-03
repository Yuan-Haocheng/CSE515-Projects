import datetime

def getDate():
    x = datetime.datetime.now()
    x = x.strftime("%b-%d-%Y_%H-%M")
    return x