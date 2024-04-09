# CommonFilters
Common Filters but now you can add your own!!!

## Usage


```python
from CommonFilters import CommonFilters
from direct.showbase.ShowBase import ShowBase
s = ShowBase()
p = s.loader.load_model("panda")
p.reparent_to(s.render)
c = CommonFilters(s.win, s.cam)
c.setBlurSharpen(0.5)

c.add_uniform("uniform float raise_amount,")
c.add_shader_input({"raise_amount": 0.5})
c.add_filter("""o_color += 0.5 * raise_amount;""", 0)

c.add_uniform("uniform float increase_red,")
c.add_shader_input({"increase_red": 0.5})
c.add_filter("""o_color.r += increase_red;""", 0)

s.run()
```
