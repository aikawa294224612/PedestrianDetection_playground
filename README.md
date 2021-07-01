# PedestrianDetection_playground
Andrew Ng的deep learning specialization系列-從底層刻一個陽春Pedestrian Detection

固定的BBox對整張image做sliding window，去判斷該BBox中是否為Pedestrian (patch classification)

細節:

1. 更改BBox大小(但input要resize) (沒做自動化)
2. 注意連續的BBox要判斷都是同一個人 (仍在思考如何改善)

**因為僅是想透過這樣從底層慢慢刻出一個以了解過程，所以沒有考慮infer速度很慢(因為要一個一個patch掃)，現今已直接透過detect一部掃整張圖(RCNN系列or YOLO)**

#### infer原圖
![Imgur](https://i.imgur.com/Tg3vmui.png)

#### training過程 (sliding window)
![Imgur](https://i.imgur.com/4tZRcrl.jpg)

#### 使用selective search
使用selective search效果fixed ROI沒有比較佳，仍須使用region proposal network (RPN)可能會比較好

![Imgur](https://i.imgur.com/cMnjwK2.jpg)

(車底有行人?危)

#### 困點 (v1.0版)
1. 連續的BBox要判斷都是同一個人，雖然已做限制，但是沒有辦法判斷是同一個人
2. 右邊海報上的雖然世人，但非行人(真人)還是會被框
3. 仍有一些其他物品(地板、紅綠燈、車尾)被判斷為行人
