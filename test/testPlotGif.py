import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

# 2.グラフ領域の作成
fig, ax = plt.subplots()
artists = []
for i in range(100):
    x = np.linspace(0, 4 * np.pi)
    y = np.sin(x - i / 100 * 2 * np.pi)
    y2 = np.cos(x -i / 100 * 2 * np.pi)

    # アニメーション化する要素の準備
    my_line, = ax.plot(x, y, "blue")
    my_text = ax.text(0, y[0], " ⇐ inlet", color="darkblue", size="large")
    my_title = ax.plot(x, y2, 'red')
    #  アニメーション化する要素をリスト化
    artists.append([my_line, my_text, my_title])

# 4. アニメーション化
anim = ArtistAnimation(fig, artists, interval = 50)
anim.save('test.gif')