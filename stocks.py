import yfinance as yf   # https://github.com/ranaroussi/yfinance

for st in ["MSFT","AAPL","GOOG","FB","AMZN","BABA","TSLA","NIO","OKTA","QCOM","AMD","IAG.L","BIDU","0700.HK"]:
    try:
        s=yf.Ticker(st).info
        print(st+"\t: "+str(round(s["bid"],1))+" / "+str(round(s["fiftyTwoWeekHigh"],1))+" (-"+str(round(100-s["bid"]/s["fiftyTwoWeekHigh"]*100,1))+"%)   peg: "+str(s["pegRatio"]))
    except:
        print(st+" not found")