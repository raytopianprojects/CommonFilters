# CommonFilters
CommonFilters but now you can add your own!!!

## How it works 
It appends the code you've added to CommonFilters internal shader string before CommonFilter compiles the shader and adds its own code.

## Usage
This version of common filters adds the methods add_filter, load_filter, add_uniform, remove_uniform, remove_filter, add_shader_input, and add_shader_inputs.

### Functions
```
add_filter(name: str, string: str, order:int = None) takes in the name of the filter (whatever you want it to be), a string and optionally the order where it should be.

load_filter(name: str, file: str, order:int = None) loads a file from disk and then see add_filter.

add_uniform(string: str) adds a uniform(s). Note you can just write type uniform_name instead of uniform type uniform_name, (ex. float 

remove_uniform(index: int) removes uniform(s) from the stored uniform list.

remove_filter(name: str, inputs: list = None, uniforms: list = None) removes the filter by name and optionally any shader inputs (by name) or uniforms (this must match the value of the string in add_uniform).

add_shader_inputs(inputs: dict) updates shader inputs

add_shader_input(name: str, value) updates shader input
```
Keep in mind I didn't port anything to glsl so you still need to use Nvidia's CG.

### Example
```python
from CommonFilters import CommonFilters
from direct.showbase.ShowBase import ShowBase

s = ShowBase()

p = s.loader.load_model("panda")
p.reparent_to(s.render)

c = CommonFilters(s.win, s.cam)
c.setBlurSharpen(0.5)

c.add_uniform("float raise_amount")
c.add_shader_inputs({"raise_amount": 0.5})
c.add_filter("""o_color += 0.5 * raise_amount;""", 0)

c.add_uniform("float increase_red")
c.add_shader_input("increase_red", 0.5})
c.add_filter("""o_color.r += increase_red;""", 0)

s.run()
```

Should look like this

![image](https://github.com/raytopianprojects/CommonFilters/assets/54505044/9ced3882-5ba6-4534-97af-749b6c459c44)


## Todo
I'll probably add support for reording the builtin filters as well but that'll be later. 
