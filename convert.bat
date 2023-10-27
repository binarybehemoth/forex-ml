@echo off

type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURUSD.input > porting\EURUSD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPUSD.input > porting\GBPUSD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDCHF.input > porting\USDCHF.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDUSD.input > porting\AUDUSD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NZDUSD.input > porting\NZDUSD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURGBP.input > porting\EURGBP.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURCAD.input > porting\EURCAD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURCHF.input > porting\EURCHF.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURAUD.input > porting\EURAUD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURNZD.input > porting\EURNZD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPCAD.input > porting\GBPCAD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPAUD.input > porting\GBPAUD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPNZD.input > porting\GBPNZD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\CADCHF.input > porting\CADCHF.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDCHF.input > porting\AUDCHF.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NZDCHF.input > porting\NZDCHF.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDNZD.input > porting\AUDNZD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\US30.input > porting\USA30IDXUSD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\US500.input > porting\USA500IDXUSD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDCAD.input > porting\USDCAD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPCHF.input > porting\GBPCHF.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDCAD.input > porting\AUDCAD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NZDCAD.input > porting\NZDCAD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\XAUUSD.input > porting\XAUUSD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDSGD.input > porting\AUDSGD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\CHFSGD.input > porting\CHFSGD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURNOK.input > porting\EURNOK.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURPLN.input > porting\EURPLN.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURSEK.input > porting\EURSEK.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURSGD.input > porting\EURSGD.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDMXN.input > porting\USDMXN.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDNOK.input > porting\USDNOK.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDPLN.input > porting\USDPLN.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDSEK.input > porting\USDSEK.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDZAR.input > porting\USDZAR.input
type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDSGD.input > porting\USDSGD.input


:start
for /f %%i in ('powershell Get-Date -Format ss') do (	
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURUSD.input > porting\EURUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPUSD.input > porting\GBPUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDCHF.input > porting\USDCHF.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDUSD.input > porting\AUDUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NZDUSD.input > porting\NZDUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURGBP.input > porting\EURGBP.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURCAD.input > porting\EURCAD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURCHF.input > porting\EURCHF.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURAUD.input > porting\EURAUD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURNZD.input > porting\EURNZD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPCAD.input > porting\GBPCAD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPAUD.input > porting\GBPAUD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPNZD.input > porting\GBPNZD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\CADCHF.input > porting\CADCHF.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDCHF.input > porting\AUDCHF.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NZDCHF.input > porting\NZDCHF.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDNZD.input > porting\AUDNZD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\US30.input > porting\USA30IDXUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\US500.input > porting\USA500IDXUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDCAD.input > porting\USDCAD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPCHF.input > porting\GBPCHF.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDCAD.input > porting\AUDCAD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NZDCAD.input > porting\NZDCAD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\XAUUSD.input > porting\XAUUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDSGD.input > porting\AUDSGD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\CHFSGD.input > porting\CHFSGD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURNOK.input > porting\EURNOK.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURPLN.input > porting\EURPLN.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURSEK.input > porting\EURSEK.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURSGD.input > porting\EURSGD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDMXN.input > porting\USDMXN.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDNOK.input > porting\USDNOK.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDPLN.input > porting\USDPLN.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDSEK.input > porting\USDSEK.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDZAR.input > porting\USDZAR.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDSGD.input > porting\USDSGD.input
		
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURUSD.input > porting\EURUSD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPUSD.input > porting\GBPUSD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDCHF.input > porting\USDCHF.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDUSD.input > porting\AUDUSD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NZDUSD.input > porting\NZDUSD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURGBP.input > porting\EURGBP.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURCAD.input > porting\EURCAD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURCHF.input > porting\EURCHF.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURAUD.input > porting\EURAUD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURNZD.input > porting\EURNZD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPCAD.input > porting\GBPCAD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPAUD.input > porting\GBPAUD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPNZD.input > porting\GBPNZD.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\CADCHF.input > porting\CADCHF.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDCHF.input > porting\AUDCHF.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NZDCHF.input > porting\NZDCHF.input
		if "%%i"=="58" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDNZD.input > porting\AUDNZD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\US30.input > porting\USA30IDXUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\US500.input > porting\USA500IDXUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDCAD.input > porting\USDCAD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GBPCHF.input > porting\GBPCHF.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDCAD.input > porting\AUDCAD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\NZDCAD.input > porting\NZDCAD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\XAUUSD.input > porting\XAUUSD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\AUDSGD.input > porting\AUDSGD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\CHFSGD.input > porting\CHFSGD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURNOK.input > porting\EURNOK.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURPLN.input > porting\EURPLN.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURSEK.input > porting\EURSEK.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\EURSGD.input > porting\EURSGD.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDMXN.input > porting\USDMXN.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDNOK.input > porting\USDNOK.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDPLN.input > porting\USDPLN.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDSEK.input > porting\USDSEK.input
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDZAR.input > porting\USDZAR.input		
		if "%%i"=="55" type c:\Users\Intel\AppData\Roaming\MetaQuotes\Terminal\Common\Files\USDSGD.input > porting\USDSGD.input

	timeout 1 /nobreak>nul
	)
goto start