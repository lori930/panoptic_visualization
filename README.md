# panoptic_visualization
A toolbox to help you explain panoptic perception models.

### Before run
```
git clone git@github.com:hustvl/YOLOP.git
pip install torchviz
```
Put the files under `YOLOP/` folder.

### Visualize your model
```
pythont visualize_model.py
```


### Generate CAM of your target layer/block
```
# example: target layer 10 (count from 0)
 python explain_yolop.py --layer 10
```
- You can also specify a block as the target layer by modifying line 75.
