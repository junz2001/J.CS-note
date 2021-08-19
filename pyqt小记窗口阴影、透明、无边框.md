# pyqt小记|窗口阴影、透明、无边框

```python
# 设置无边框圆角带阴影窗口
self.MainWindow.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 无边框
# ===============透明阴影====================
self.MainWindow.setAutoFillBackground(True) #一定要加上
self.MainWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 窗口透明
shadow=QGraphicsDropShadowEffect()  # 创建阴影
shadow.setBlurRadius(20)  # 设置阴影大小为9px
shadow.setColor(QColor("#444444"))  # 设置颜色透明度为100的（0,0,0）黑色
shadow.setOffset(0,0)  # 阴影偏移距离为0px
self.MainWindow.setGraphicsEffect(shadow)  # 添加阴影
self.MainWindow.resize(1600, 900)
```

