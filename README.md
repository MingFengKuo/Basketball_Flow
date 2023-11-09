# Basketball Flow

### Basketball Flow: Learning to Synthesize Realistic and Diverse Basketball GamePlays based on Tactical Sketches

Ming-Feng Kuo<sup>1</sup>, Yu-Shuen Wang<sup>1</sup>

<sup>1</sup>National Yang Ming Chiao Tung University

> This article introduces "Basketball Flow," a system designed to create diverse and realistic basketball game simulations based on predetermined strategy sketches. By providing a strategy sketch, the model generates various offensive and defensive player simulations within the specified sketch. Such simulations aid less experienced players in understanding strategies and potential challenges more intuitively. To achieve this, Basketball Flow combines common generative models, leveraging their strengths and compensating for their weaknesses. Compared to previous research, Basketball Flow offers more realistic and diverse basketball gameplay and consistently receives higher ratings across various basketball gameplay features, affirming its effectiveness through experimental results.

### Visualizing Basketball Gameplays

<img src="https://github.com/MingFengKuo/Basketball_Flow/blob/main/Image/diversity_demo.jpg" title="Visualizing Basketball Gameplays" height=60% width=60%/>

> Tactical sketches at the top, followed by model-generated game scenarios in rows 'a' to 'c.’ Columns 1 to 4 represent game phases with green, red, and blue dots indicating the ball, offensive players, and defensive players, respectively.

### Diversity Comparison

<img src="https://github.com/MingFengKuo/Basketball_Flow/blob/main/Image/diversity.jpg" title="Diversity Comparison" height=60% width=60%/>

> We compare the diversity of simulated game plays by overlaying 100 generated trajectories of a particular player under the same conditions. Black tracks represent input conditions, while green, red, and blue tracks correspond to basketball, offensive player, and defensive player movements. Among these, Basketball Flow demonstrates the highest diversity.

### Download

[Dataset](https://drive.google.com/drive/folders/1Kcf0lA0qrHvuIhAsZw2ia83s1jzPcE0r?usp=sharing)

[Checkpoints](https://drive.google.com/drive/folders/1ibwMegJEvM25YcQg_N3b-bnSkZQmixHG?usp=sharing)

```
Basketball_Flow/
├── data/[Dataset]
├── checkpoints/[Checkpoints]
└── ...
```

### Training

```sh
~/$ git clone https://github.com/MingFengKuo/Basketball_Flow
~/$ cd Basketball_Flow
~/Basketball_Flow$ python train.py
```

The default output directory is `log`.

### Testing

```sh
~/Basketball_Flow$ python test.py
```

The default output directory is `checkpoints`.
