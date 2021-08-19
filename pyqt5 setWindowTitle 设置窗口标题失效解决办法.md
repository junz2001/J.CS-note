# pyqt5 setWindowTitle 设置窗口标题失效解决办法

```python
myMainWindow = QMainWindow()
myUi = wMain.Ui_MainWindow()
myUi.setupUi(myMainWindow)
myMainWindow.setWindowTitle('ZDEMViewer    离散元数值模拟可视化程序')
myMainWindow.setWindowIcon(QIcon("./icons/logo.ico"))
myMainWindow.show()
```

如上所示的设置顺序可以成功

<img src=".\res\pyqt\2021-08-19.png" style="zoom: 80%;" />

如果是：

```python
myMainWindow = QMainWindow()
myMainWindow.setWindowTitle('ZDEMViewer    离散元数值模拟可视化程序')
myMainWindow.setWindowIcon(QIcon("./icons/logo.ico"))
myUi = wMain.Ui_MainWindow()
myUi.setupUi(myMainWindow)
myMainWindow.show()
```

会出现**图标能够生效**，但是标题在生效之后一闪而过，变为“MainWindow”

<img src=".\res\pyqt\2021-08-19 112919.png" style="zoom:80%;" />