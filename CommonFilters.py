"""
Class CommonFilters implements certain common image
postprocessing filters.  See the :ref:`common-image-filters` page for
more information about how to use these filters.

These filters are written in the Cg shading language.
"""

# It is not ideal that these filters are all included in a single
# monolithic module.  Unfortunately, when you want to apply two filters
# at the same time, you have to compose them into a single shader, and
# the composition process isn't simply a question of concatenating them:
# you have to somehow make them work together.  I suspect that there
# exists some fairly simple framework that would make this automatable.
# However, until I write some more filters myself, I won't know what
# that framework is.  Until then, I'll settle for this
# clunky approach.  - Josh
from panda3d.core import VirtualFileSystem

vfs: VirtualFileSystem = VirtualFileSystem.get_global_ptr()
from panda3d.core import LVecBase4, LPoint2
from panda3d.core import AuxBitplaneAttrib, AntialiasAttrib
from panda3d.core import Texture, Shader, ATSNone
from panda3d.core import FrameBufferProperties
from panda3d.core import getDefaultCoordinateSystem, CS_zup_right, CS_zup_left

from direct.task.TaskManagerGlobal import taskMgr

from direct.filter.FilterManager import FilterManager
from direct.filter.filterBloomI import BLOOM_I
from direct.filter.filterBloomX import BLOOM_X
from direct.filter.filterBloomY import BLOOM_Y
from direct.filter.filterBlurX import BLUR_X
from direct.filter.filterBlurY import BLUR_Y
from direct.filter.filterCopy import COPY
from direct.filter.filterDown4 import DOWN_4

CARTOON_BODY = """
float4 cartoondelta = k_cartoonseparation * texpix_txaux.xwyw;
float4 cartoon_c0 = tex2D(k_txaux, %(texcoord)s + cartoondelta.xy);
float4 cartoon_c1 = tex2D(k_txaux, %(texcoord)s - cartoondelta.xy);
float4 cartoon_c2 = tex2D(k_txaux, %(texcoord)s + cartoondelta.wz);
float4 cartoon_c3 = tex2D(k_txaux, %(texcoord)s - cartoondelta.wz);
float4 cartoon_mx = max(cartoon_c0, max(cartoon_c1, max(cartoon_c2, cartoon_c3)));
float4 cartoon_mn = min(cartoon_c0, min(cartoon_c1, min(cartoon_c2, cartoon_c3)));
float cartoon_thresh = saturate(dot(cartoon_mx - cartoon_mn, float4(3,3,0,0)) - 0.5);
o_color = lerp(o_color, k_cartooncolor, cartoon_thresh);
"""

# Some GPUs do not support variable-length loops.
#
# We fill in the actual value of numsamples in the loop limit
# when the shader is configured.
#
SSAO_BODY = """//Cg

void vshader(float4 vtx_position : POSITION,
             float2 vtx_texcoord : TEXCOORD0,
             out float4 l_position : POSITION,
             out float2 l_texcoord : TEXCOORD0,
             out float2 l_texcoordD : TEXCOORD1,
             out float2 l_texcoordN : TEXCOORD2,
             uniform float4 texpad_depth,
             uniform float4 texpad_normal,
             uniform float4x4 mat_modelproj)
{
  l_position = mul(mat_modelproj, vtx_position);
  l_texcoord = vtx_texcoord;
  l_texcoordD = vtx_texcoord * texpad_depth.xy * 2;
  l_texcoordN = vtx_texcoord * texpad_normal.xy * 2;
}

float3 sphere[16] = float3[](float3(0.53812504, 0.18565957, -0.43192),float3(0.13790712, 0.24864247, 0.44301823),float3(0.33715037, 0.56794053, -0.005789503),float3(-0.6999805, -0.04511441, -0.0019965635),float3(0.06896307, -0.15983082, -0.85477847),float3(0.056099437, 0.006954967, -0.1843352),float3(-0.014653638, 0.14027752, 0.0762037),float3(0.010019933, -0.1924225, -0.034443386),float3(-0.35775623, -0.5301969, -0.43581226),float3(-0.3169221, 0.106360726, 0.015860917),float3(0.010350345, -0.58698344, 0.0046293875),float3(-0.08972908, -0.49408212, 0.3287904),float3(0.7119986, -0.0154690035, -0.09183723),float3(-0.053382345, 0.059675813, -0.5411899),float3(0.035267662, -0.063188605, 0.54602677),float3(-0.47761092, 0.2847911, -0.0271716));

void fshader(out float4 o_color : COLOR,
             uniform float4 k_params1,
             uniform float4 k_params2,
             float2 l_texcoord : TEXCOORD0,
             float2 l_texcoordD : TEXCOORD1,
             float2 l_texcoordN : TEXCOORD2,
             uniform sampler2D k_random : TEXUNIT0,
             uniform sampler2D k_depth : TEXUNIT1,
             uniform sampler2D k_normal : TEXUNIT2)
{
  float pixel_depth = tex2D(k_depth, l_texcoordD).a;
  float3 pixel_normal = (tex2D(k_normal, l_texcoordN).xyz * 2.0 - 1.0);
  float3 random_vector = normalize((tex2D(k_random, l_texcoord * 18.0 + pixel_depth + pixel_normal.xy).xyz * 2.0) - float3(1.0)).xyz;
  float occlusion = 0.0;
  float radius = k_params1.z / pixel_depth;
  float depth_difference;
  float3 sample_normal;
  float3 ray;
  for(int i = 0; i < %d; ++i) {
   ray = radius * reflect(sphere[i], random_vector);
   sample_normal = (tex2D(k_normal, l_texcoordN + ray.xy).xyz * 2.0 - 1.0);
   depth_difference =  (pixel_depth - tex2D(k_depth,l_texcoordD + ray.xy).r);
   occlusion += step(k_params2.y, depth_difference) * (1.0 - dot(sample_normal.xyz, pixel_normal)) * (1.0 - smoothstep(k_params2.y, k_params2.x, depth_difference));
  }
  o_color.rgb = 1.0 + (occlusion * k_params1.y);
  o_color.a = 1.0;
}
"""


class FilterConfig:
    pass


class CommonFilters:
    """ Class CommonFilters implements certain common image postprocessing
    filters.  The constructor requires a filter builder as a parameter. """

    def __init__(self, win, cam):
        self.finalQuad = None
        self.texcoords = {}
        self.texcoordPadding = {}
        self.textures = {}

        self.current_text = None
        self.manager = FilterManager(win, cam)
        self.configuration = {}
        self.task = None
        self.cleanup()
        self.filters = {}
        self.uniforms = []
        self.shader_inputs = {}
        self.render_textures = {}
        self.auxbits = 0
        self.fbprops = FrameBufferProperties()

    def cleanup(self):
        self.manager.cleanup()
        self.finalQuad = None
        self.blur = []
        self.ssao = []
        self.render_textures = {}

        if self.task is not None:
            taskMgr.remove(self.task)
            self.task = None

    def load_filter(self, name, shader_string,
                    uniforms=None,
                    shader_inputs=None,
                    needed_textures=None,
                    needed_coords=None,
                    render_into=None,
                    auxbits=None,
                    is_filepath=False,
                    order=None):

        if uniforms:
            for uniform in uniforms:
                self.add_uniform(uniform, reconfigure=False)

        if is_filepath:
            if order:
                self.filters[name] = [vfs.get_file(shader_string), order]
            else:
                self.filters[name] = [vfs.get_file(shader_string), len(self.filters)]
        else:
            if order:
                self.filters[name] = [shader_string, order]
            else:
                self.filters[name] = [shader_string, len(self.filters)]

        self.reconfigure(True, None, needed_textures=needed_textures, needed_coords=needed_coords,
                         render_into=render_into, auxbits=auxbits)

        if shader_inputs:
            self.set_shader_inputs(shader_inputs)

    def add_uniform(self, string, reconfigure=True):
        self.uniforms.append(string)
        if reconfigure:
            self.reconfigure(True, None)

    def del_uniforms(self, uniforms, reconfigure=True):
        for uniform in uniforms:
            if uniform in self.uniforms:
                self.uniforms.remove(uniform)
        self.reconfigure(reconfigure, None)

    def del_filter(self, name, inputs=None, uniforms=None):
        if name in self.filters:
            del self.filters[name]
        if inputs:
            self.del_shader_inputs(inputs, reconfigure=False)
        if uniforms:
            self.del_uniforms(uniforms, reconfigure=False)

        self.reconfigure(True, None)

    def set_shader_inputs(self, inputs):
        self.shader_inputs.update(inputs)
        self.finalQuad.setShaderInputs(**inputs)

    def set_shader_input(self, name, value):
        self.shader_inputs[name] = value
        self.finalQuad.setShaderInput(name, value)

    def del_shader_inputs(self, inputs, reconfigure=True):
        for input in inputs:
            if input in self.shader_inputs:
                del self.shader_inputs[input]
        self.reconfigure(reconfigure, None)

    def reconfigure(self, fullrebuild, changed, needed_textures=None, needed_coords=None, render_into=None,
                    auxbits=None):
        """ Reconfigure is called whenever any configuration change is made. """
        configuration = self.configuration

        self.cleanup()

        if not self.manager.win.gsg.getSupportsBasicShaders():
            return False

        self.auxbits = 0

        if auxbits:
            for auxbit in auxbits:
                self.auxbits |= auxbit

        needtex = {"color"}
        needtexcoord = {"color"}

        if needed_textures:
            for texture in needed_textures:
                needtex.add(texture)

        if needed_coords:
            for coords in needed_coords:
                needtexcoord.add(coords)

        if "AmbientOcclusion" in configuration:
            needtex.add("depth")
            needtex.add("ssao0")
            needtex.add("ssao1")
            needtex.add("ssao2")
            needtex.add("aux")
            self.auxbits |= AuxBitplaneAttrib.ABOAuxNormal
            needtexcoord.add("ssao2")

        if "ViewGlow" in configuration:
            self.auxbits |= AuxBitplaneAttrib.ABOGlow

        if "VolumetricLighting" in configuration:
            needtex.add(configuration["VolumetricLighting"].source)

        for tex in needtex:
            self.textures[tex] = Texture("scene-" + tex)
            self.textures[tex].setWrapU(Texture.WMClamp)
            self.textures[tex].setWrapV(Texture.WMClamp)

        fbprops = None
        clamping = None
        if "HighDynamicRange" in configuration:
            fbprops = FrameBufferProperties()
            fbprops.setFloatColor(True)
            fbprops.setSrgbColor(False)
            clamping = False

        self.finalQuad = self.manager.renderSceneInto(textures=self.textures, auxbits=self.auxbits, fbprops=fbprops,
                                                      clamping=clamping)

        if self.finalQuad is None:
            self.cleanup()
            return False

        if render_into:
            for render_name, settings in render_into.items():

                mul = settings.get("mul")
                if not mul:
                    mul = 1

                div = settings.get("div")
                if not div:
                    div = 1

                align = settings.get("align")
                if not align:
                    align = 1

                depthtex = settings.get("depthtex")
                if type(depthtex) is dict:
                    depthtex = self.textures[depthtex["texture"]]

                colortex = settings.get("colortex")
                if type(colortex) is dict:
                    colortex = self.textures[colortex["texture"]]

                auxtex0 = settings.get("auxtex0")
                if type(auxtex0) is dict:
                    auxtex0 = self.textures[auxtex0["texture"]]

                auxtex1 = settings.get("auxtex1")
                if type(auxtex1) is dict:
                    auxtex1 = self.textures[auxtex1["texture"]]

                fbprops = settings.get("fbprops")

                quad = self.manager.renderQuadInto(render_name,
                                                   mul=mul,
                                                   align=align,
                                                   div=div,
                                                   depthtex=depthtex,
                                                   colortex=colortex,
                                                   auxtex0=auxtex0,
                                                   auxtex1=auxtex1,
                                                   fbprops=fbprops)

                self.render_textures[render_name] = quad

                shader = settings.get("shader")
                if shader:
                    quad.set_shader(shader)

                shader_inputs = settings.get("shader_inputs")
                if shader_inputs:
                    for name, value in shader_inputs.items():
                        if type(value) is dict:
                            if "texture" in value:
                                value = self.textures[value["texture"]]
                        quad.set_shader_input(name, value)

        if "AmbientOcclusion" in configuration:
            ssao0 = self.textures["ssao0"]
            ssao1 = self.textures["ssao1"]
            ssao2 = self.textures["ssao2"]
            self.ssao.append(self.manager.renderQuadInto("filter-ssao0", colortex=ssao0))
            self.ssao.append(self.manager.renderQuadInto("filter-ssao1", colortex=ssao1, div=2))
            self.ssao.append(self.manager.renderQuadInto("filter-ssao2", colortex=ssao2))
            self.ssao[0].setShaderInput("depth", self.textures["depth"])
            self.ssao[0].setShaderInput("normal", self.textures["aux"])
            self.ssao[0].setShaderInput("random", base.loader.loadTexture("maps/random.rgb"))
            self.ssao[0].setShader(
                Shader.make(SSAO_BODY % configuration["AmbientOcclusion"].numsamples, Shader.SL_Cg))
            self.ssao[1].setShaderInput("src", ssao0)
            self.ssao[1].setShader(Shader.make(BLUR_X, Shader.SL_Cg))
            self.ssao[2].setShaderInput("src", ssao1)
            self.ssao[2].setShader(Shader.make(BLUR_Y, Shader.SL_Cg))

        print(needtexcoord, "Needtexcoord")
        for tex in needtexcoord:
            if self.textures[tex].getAutoTextureScale() != ATSNone or "HalfPixelShift" in configuration:
                self.texcoords[tex] = "l_texcoord_" + tex
                self.texcoordPadding["l_texcoord_" + tex] = tex
            else:
                # Share unpadded texture coordinates.
                self.texcoords[tex] = "l_texcoord"
                self.texcoordPadding["l_texcoord"] = None

        print(self.texcoords, "self.texcoords")
        texcoordSets = list(enumerate(self.texcoordPadding.keys()))

        text = "//Cg\n"

        if "HighDynamicRange" in configuration:
            text += """static const float3x3 aces_input_mat = {
                      {0.59719, 0.35458, 0.04823},
                      {0.07600, 0.90834, 0.01566},
                      {0.02840, 0.13383, 0.83777},
                    };
                    static const float3x3 aces_output_mat = {
                      { 1.60475, -0.53108, -0.07367},
                      {-0.10208,  1.10813, -0.00605},
                      {-0.00327, -0.07276,  1.07602},
                    };"""

        text += "void vshader(float4 vtx_position : POSITION,\n"
        text += "  out float4 l_position : POSITION,\n"
        text += "  out float4 l_fragpos ,\n"

        for texcoord, padTex in self.texcoordPadding.items():
            if padTex is not None:
                text += "  uniform float4 texpad_tx%s,\n" % (padTex)
                if "HalfPixelShift" in configuration:
                    text += "  uniform float4 texpix_tx%s,\n" % (padTex)

        for i, name in texcoordSets:
            text += "  out float2 %s : TEXCOORD%d,\n" % (name, i)

        text += "  uniform float4x4 mat_modelproj)\n"
        text += "{\n"
        text += "  l_position = mul(mat_modelproj, vtx_position);\n"

        # The card is oriented differently depending on our chosen
        # coordinate system.  We could just use vtx_texcoord, but this
        # saves on an additional variable.
        if getDefaultCoordinateSystem() in (CS_zup_right, CS_zup_left):
            pos = "vtx_position.xz"
        else:
            pos = "vtx_position.xy"

        for texcoord, padTex in self.texcoordPadding.items():
            if padTex is None:
                text += "  %s = %s * float2(0.5, 0.5) + float2(0.5, 0.5);\n" % (texcoord, pos)
            else:
                text += "  %s = (%s * texpad_tx%s.xy) + texpad_tx%s.xy;\n" % (texcoord, pos, padTex, padTex)

                if "HalfPixelShift" in configuration:
                    text += "  %s += texpix_tx%s.xy * 0.5;\n" % (texcoord, padTex)

        text += "l_fragpos = l_position;"
        text += "}\n"

        text += "void fshader(\n"
        text += "float4 l_fragpos,\n"

        for i, name in texcoordSets:
            text += "  float2 %s : TEXCOORD%d,\n" % (name, i)

        for key in self.textures:
            text += "  uniform sampler2D k_tx" + key + ",\n"

        if "VolumetricLighting" in configuration:
            text += "  uniform float4 k_casterpos,\n"
            text += "  uniform float4 k_vlparams,\n"

        if "ExposureAdjust" in configuration:
            text += "  uniform float k_exposure,\n"

        for uniform in self.uniforms:
            text += f"uniform {uniform},\n"

        text += "  out float4 o_color : COLOR)\n"
        text += "{\n"
        text += """        l_fragpos /= l_fragpos.w; l_fragpos.xy = (l_fragpos.xy + 1) / 2;"""
        text += "  o_color = tex2D(k_txcolor, %s);\n" % (self.texcoords["color"])

        user_filters = sorted(self.filters.values(), key=lambda l: l[1])
        for user_filter in user_filters:
            text += user_filter[0]

        if "AmbientOcclusion" in configuration:
            text += "  o_color *= tex2D(k_txssao2, %s).r;\n" % (self.texcoords["ssao2"])
        if "ViewGlow" in configuration:
            text += "  o_color.r = o_color.a;\n"
        if "VolumetricLighting" in configuration:
            text += "  float decay = 1.0f;\n"
            text += "  float2 curcoord = %s;\n" % (self.texcoords["color"])
            text += "  float2 lightdir = curcoord - k_casterpos.xy;\n"
            text += "  lightdir *= k_vlparams.x;\n"
            text += "  half4 sample = tex2D(k_txcolor, curcoord);\n"
            text += "  float3 vlcolor = sample.rgb * sample.a;\n"
            text += "  for (int i = 0; i < %s; i++) {\n" % (int(configuration["VolumetricLighting"].numsamples))
            text += "    curcoord -= lightdir;\n"
            text += "    sample = tex2D(k_tx%s, curcoord);\n" % (configuration["VolumetricLighting"].source)
            text += "    sample *= sample.a * decay;//*weight\n"
            text += "    vlcolor += sample.rgb;\n"
            text += "    decay *= k_vlparams.y;\n"
            text += "  }\n"
            text += "  o_color += float4(vlcolor * k_vlparams.z, 1);\n"

        if "ExposureAdjust" in configuration:
            text += "  o_color.rgb *= k_exposure;\n"

        # With thanks to Stephen Hill!
        if "HighDynamicRange" in configuration:
            text += "  float3 aces_color = mul(aces_input_mat, o_color.rgb);\n"
            text += "  o_color.rgb = saturate(mul(aces_output_mat, (aces_color * (aces_color + 0.0245786f) - 0.000090537f) / (aces_color * (0.983729f * aces_color + 0.4329510f) + 0.238081f)));\n"

        if "GammaAdjust" in configuration:
            gamma = configuration["GammaAdjust"]
            if gamma == 0.5:
                text += "  o_color.rgb = sqrt(o_color.rgb);\n"
            elif gamma == 2.0:
                text += "  o_color.rgb *= o_color.rgb;\n"
            elif gamma != 1.0:
                text += "  o_color.rgb = pow(o_color.rgb, %ff);\n" % (gamma)

        if "SrgbEncode" in configuration:
            text += "  o_color.r = (o_color.r < 0.0031308) ? (o_color.r * 12.92) : (1.055 * pow(o_color.r, 0.41666) - 0.055);\n"
            text += "  o_color.g = (o_color.g < 0.0031308) ? (o_color.g * 12.92) : (1.055 * pow(o_color.g, 0.41666) - 0.055);\n"
            text += "  o_color.b = (o_color.b < 0.0031308) ? (o_color.b * 12.92) : (1.055 * pow(o_color.b, 0.41666) - 0.055);\n"

        text += "}\n"

        self.current_text = text

        shader = Shader.make(text, Shader.SL_Cg)
        if not shader:
            return False
        self.finalQuad.setShader(shader)
        for tex in self.textures:
            self.finalQuad.setShaderInput("tx" + tex, self.textures[tex])

        self.task = taskMgr.add(self.update, "common-filters-update")

        if changed == "VolumetricLighting" or fullrebuild:
            if "VolumetricLighting" in configuration:
                config = configuration["VolumetricLighting"]
                tcparam = config.density / float(config.numsamples)
                self.finalQuad.setShaderInput("vlparams", tcparam, config.decay, config.exposure, 0.0)

        if changed == "AmbientOcclusion" or fullrebuild:
            if "AmbientOcclusion" in configuration:
                config = configuration["AmbientOcclusion"]
                self.ssao[0].setShaderInput("params1", config.numsamples, -float(config.amount) / config.numsamples,
                                            config.radius, 0)
                self.ssao[0].setShaderInput("params2", config.strength, config.falloff, 0, 0)

        if changed == "ExposureAdjust" or fullrebuild:
            if "ExposureAdjust" in configuration:
                stops = configuration["ExposureAdjust"]
                self.finalQuad.setShaderInput("exposure", 2 ** stops)

        self.finalQuad.setShaderInputs(**self.shader_inputs)

        self.update()
        return True

    def update(self, task=None):
        """Updates the shader inputs that need to be updated every frame.
        Normally, you shouldn't call this, it's being called in a task."""

        if "VolumetricLighting" in self.configuration:
            caster = self.configuration["VolumetricLighting"].caster
            casterpos = LPoint2()
            self.manager.camera.node().getLens().project(caster.getPos(self.manager.camera), casterpos)
            self.finalQuad.setShaderInput("casterpos",
                                          LVecBase4(casterpos.getX() * 0.5 + 0.5, (casterpos.getY() * 0.5 + 0.5), 0, 0))
        if task is not None:
            return task.cont

    def setMSAA(self, samples):
        """Enables multisample anti-aliasing on the render-to-texture buffer.
        If you enable this, it is recommended to leave any multisample request
        on the main framebuffer OFF (ie. don't set framebuffer-multisample true
        in Config.prc), since it would be a waste of resources otherwise.

        .. versionadded:: 1.10.13
        """

        self.fbprops.setMultisamples(samples)

        camNode = self.manager.camera.node()
        state = camNode.getInitialState()
        state.setAttrib(AntialiasAttrib.make(AntialiasAttrib.M_multisample))
        camNode.setInitialState(state)

    def delMSAA(self):
        self.fbprops.setMultisamples(0)
        # Somehow remove fbprop from final quad

    def setCartoonInk(self, separation=1, color=(0, 0, 0, 1)):
        self.load_filter(
            "CartoonInk",
            """float4 cartoondelta = k_cartoonseparation * texpix_txaux.xwyw;
float4 cartoon_c0 = tex2D(k_txaux, l_texcoord_aux + cartoondelta.xy);
float4 cartoon_c1 = tex2D(k_txaux, l_texcoord_aux - cartoondelta.xy);
float4 cartoon_c2 = tex2D(k_txaux, l_texcoord_aux + cartoondelta.wz);
float4 cartoon_c3 = tex2D(k_txaux, l_texcoord_aux - cartoondelta.wz);
float4 cartoon_mx = max(cartoon_c0, max(cartoon_c1, max(cartoon_c2, cartoon_c3)));
float4 cartoon_mn = min(cartoon_c0, min(cartoon_c1, min(cartoon_c2, cartoon_c3)));
float cartoon_thresh = saturate(dot(cartoon_mx - cartoon_mn, float4(3,3,0,0)) - 0.5);
o_color = lerp(o_color, k_cartooncolor, cartoon_thresh);""",
            uniforms=["float4 k_cartoonseparation",
                      "float4 k_cartooncolor",
                      "float4 texpix_txaux"],
            auxbits=[AuxBitplaneAttrib.ABOAuxNormal],
            needed_textures=["aux"],
            needed_coords=["aux"], order=3, shader_inputs={"cartoonseparation": separation, "cartooncolor": color}
        )

        # Uniforms cartoonseparation, cartooncolor

    def delCartoonInk(self):
        self.del_filter("CartoonInk", uniforms=["float4 k_cartoonseparation",
                      "float4 k_cartooncolor",
                      "float4 texpix_txaux"], )

    def setBloom(self, blend=(0.3, 0.4, 0.3, 0.0), mintrigger=0.6, maxtrigger=1.0, desat=0.6, intensity=1.0,
                 size="medium"):
        """
        Applies the Bloom filter to the output.
        size can either be "off", "small", "medium", or "large".
        Setting size to "off" will remove the Bloom filter.
        """
        if size == 0 or size == "off":
            self.delBloom()
            return
        elif size == 1:
            size = "small"
        elif size == 2:
            size = "medium"
        elif size == 3:
            size = "large"

        if maxtrigger is None:
            maxtrigger = mintrigger + 0.8

        intensity *= 3.0
        if size == "large":
            scale = 8
            downsamplerName = "filter-down4"
            downsampler = DOWN_4
        elif size == "medium":
            scale = 4
            downsamplerName = "filter-copy"
            downsampler = COPY
        else:
            scale = 2
            downsamplerName = "filter-copy"
            downsampler = COPY

        self.load_filter("Bloom",
                         """
        o_color = saturate(o_color);
        float4 bloom = 0.5 * tex2D(k_txbloom3, l_texcoord_bloom3);
        o_color = 1-((1-bloom)*(1-o_color));
        """,
                         auxbits=[AuxBitplaneAttrib.ABOGlow],
                         needed_coords=["bloom3"],
                         needed_textures=["bloom0", "bloom1", "bloom2", "bloom3"],
                         render_into={
                             "filter-bloomi":
                                 {
                                     "shader_inputs":
                                         {
                                             "blend": (blend[0], blend[1], blend[2], blend[3] * 2.0),
                                             "trigger": (mintrigger, 1.0 / (maxtrigger - mintrigger), 0.0, 0.0),
                                             "desat": desat,
                                             "src": {"texture": "color"}
                                         },
                                     "shader": Shader.make(BLOOM_I, Shader.SL_Cg),
                                     "colortex": {"texture": "bloom0"},
                                     "div": 2,
                                     "align": scale

                                 },
                             downsamplerName:
                                 {
                                     "colortex": {"texture": "bloom1"},
                                     "div": scale,
                                     "align": scale,
                                     "shader": Shader.make(downsampler, Shader.SL_Cg),
                                     "shader_inputs":
                                         {
                                             "src": {"texture": "bloom0"}
                                         }

                                 },
                             "filter-bloomx":
                                 {
                                     "colortex": {"texture": "bloom2"},
                                     "div": scale,
                                     "align": scale,
                                     "shader": Shader.make(BLOOM_X, Shader.SL_Cg),
                                     "shader_inputs":
                                         {
                                             "src": {"texture": "bloom1"}
                                         }

                                 },
                             "filter-bloomy":
                                 {
                                     "colortex": {"texture": "bloom3"},
                                     "div": scale,
                                     "align": scale,
                                     "shader": Shader.make(BLOOM_Y, Shader.SL_Cg),

                                     "shader_inputs":
                                         {
                                             "intensity": (intensity, intensity, intensity, intensity),
                                             "src": {"texture": "bloom2"}
                                         }
                                 }
                         }
                         )

    def delBloom(self):
        self.del_filter("Bloom")

    def setHalfPixelShift(self):
        fullrebuild = ("HalfPixelShift" not in self.configuration)
        self.configuration["HalfPixelShift"] = 1
        return self.reconfigure(fullrebuild, "HalfPixelShift")

    def delHalfPixelShift(self):
        if "HalfPixelShift" in self.configuration:
            del self.configuration["HalfPixelShift"]
            return self.reconfigure(True, "HalfPixelShift")
        return True

    def setViewGlow(self):
        fullrebuild = ("ViewGlow" not in self.configuration)
        self.configuration["ViewGlow"] = 1
        return self.reconfigure(fullrebuild, "ViewGlow")

    def delViewGlow(self):
        if "ViewGlow" in self.configuration:
            del self.configuration["ViewGlow"]
            return self.reconfigure(True, "ViewGlow")
        return True

    def setInverted(self):
        self.load_filter("Inverted", "  o_color = float4(1, 1, 1, 1) - o_color;\n", order=14)

        return self.reconfigure(True, "Inverted")

    def delInverted(self):
        self.del_filter("Inverted")

    def setVolumetricLighting(self, caster, numsamples=32, density=5.0, decay=0.1, exposure=0.1, source="color"):
        oldconfig = self.configuration.get("VolumetricLighting", None)
        fullrebuild = True
        if oldconfig and oldconfig.source == source and oldconfig.numsamples == int(numsamples):
            fullrebuild = False
        newconfig = FilterConfig()
        newconfig.caster = caster
        newconfig.numsamples = int(numsamples)
        newconfig.density = density
        newconfig.decay = decay
        newconfig.exposure = exposure
        newconfig.source = source
        self.configuration["VolumetricLighting"] = newconfig
        return self.reconfigure(fullrebuild, "VolumetricLighting")

    def delVolumetricLighting(self):
        if "VolumetricLighting" in self.configuration:
            del self.configuration["VolumetricLighting"]
            return self.reconfigure(True, "VolumetricLighting")
        return True

    def setBlurSharpen(self, amount=0.0):
        """Enables the blur/sharpen filter. If the 'amount' parameter is 1.0, it will not have any effect.
        A value of 0.0 means fully blurred, and a value higher than 1.0 sharpens the image."""

        self.load_filter(
            "BlurSharpen",
            "  o_color = lerp(tex2D(k_txblur1, l_texcoord_blur1), o_color, k_blurval.x);\n",
            shader_inputs={"blurval": LVecBase4(amount, amount, amount, amount)},
            uniforms=["float4 k_blurval"],
            needed_textures=["blur0", "blur1"],
            needed_coords=["blur1"],
            render_into={
                "filter-blur0": {
                    "colortex": {"texture": "blur0"},
                    "div": 2,
                    "shader_inputs": {
                        "src": {"texture": "color"}
                    },
                    "shader": Shader.make(BLUR_X, Shader.SL_Cg)
                },
                "filter-blur1": {
                    "colortex": {"texture": "blur1"},
                    "shader_inputs":
                        {
                            "src": {"texture": "blur0"}
                        },
                    "shader": Shader.make(BLUR_Y, Shader.SL_Cg)
                }
            }
        )

    def delBlurSharpen(self):
        self.del_filter("BlurSharpen", inputs=["blurval"], uniforms=["float4 k_blurval"])

    def setAmbientOcclusion(self, numsamples=16, radius=0.05, amount=2.0, strength=0.01, falloff=0.000002):
        fullrebuild = ("AmbientOcclusion" not in self.configuration)

        if not fullrebuild:
            fullrebuild = (numsamples != self.configuration["AmbientOcclusion"].numsamples)

        newconfig = FilterConfig()
        newconfig.numsamples = numsamples
        newconfig.radius = radius
        newconfig.amount = amount
        newconfig.strength = strength
        newconfig.falloff = falloff
        self.configuration["AmbientOcclusion"] = newconfig
        return self.reconfigure(fullrebuild, "AmbientOcclusion")

    def delAmbientOcclusion(self):
        if "AmbientOcclusion" in self.configuration:
            del self.configuration["AmbientOcclusion"]
            return self.reconfigure(True, "AmbientOcclusion")
        return True

    def setGammaAdjust(self, gamma):
        """ Applies additional gamma correction to the image.  1.0 = no correction. """

        if gamma == 0.5:
            self.load_filter(
                "GammaAdjust",
                "  o_color.rgb = sqrt(o_color.rgb);\n"
            )

        elif gamma == 2.0:
            self.load_filter(
                "GammaAdjust",
                "  o_color.rgb *= o_color.rgb;\n"
            )

        elif gamma != 1.0:
            self.load_filter(
                "GammaAdjust",
                "  o_color.rgb = pow(o_color.rgb, %ff);\n" % (gamma)
            )

        return self.reconfigure(True, "GammaAdjust")

    def delGammaAdjust(self):
        self.del_filter("GammaAdjust")

    def setSrgbEncode(self, force=False):
        """ Applies the inverse sRGB EOTF to the output, unless the window
        already has an sRGB framebuffer, in which case this filter refuses to
        apply, to prevent accidental double-application.

        Set the force argument to True to force it to be applied in all cases.

        .. versionadded:: 1.10.7
        """
        new_enable = force or not self.manager.win.getFbProperties().getSrgbColor()
        old_enable = self.configuration.get("SrgbEncode", False)
        if new_enable and not old_enable:
            self.configuration["SrgbEncode"] = True
            return self.reconfigure(True, "SrgbEncode")
        elif not new_enable and old_enable:
            del self.configuration["SrgbEncode"]
        return new_enable

    def delSrgbEncode(self):
        """ Reverses the effects of setSrgbEncode. """
        if "SrgbEncode" in self.configuration:
            old_enable = self.configuration["SrgbEncode"]
            del self.configuration["SrgbEncode"]
            return self.reconfigure(old_enable, "SrgbEncode")
        return True

    def setHighDynamicRange(self):
        """ Enables HDR rendering by using a floating-point framebuffer,
        disabling color clamping on the main scene, and applying a tone map
        operator (ACES).

        It may also be necessary to use setExposureAdjust to perform exposure
        compensation on the scene, depending on the lighting intensity.

        .. versionadded:: 1.10.7
        """

        fullrebuild = (("HighDynamicRange" in self.configuration) is False)
        self.configuration["HighDynamicRange"] = 1
        return self.reconfigure(fullrebuild, "HighDynamicRange")

    def delHighDynamicRange(self):
        if "HighDynamicRange" in self.configuration:
            del self.configuration["HighDynamicRange"]
            return self.reconfigure(True, "HighDynamicRange")
        return True

    def setExposureAdjust(self, stops):
        """ Sets a relative exposure adjustment to multiply with the result of
        rendering the scene, in stops.  A value of 0 means no adjustment, a
        positive value will result in a brighter image.  Useful in conjunction
        with HDR, see setHighDynamicRange.

        .. versionadded:: 1.10.7
        """
        old_stops = self.configuration.get("ExposureAdjust")
        if old_stops != stops:
            self.configuration["ExposureAdjust"] = stops
            return self.reconfigure(old_stops is None, "ExposureAdjust")
        return True

    def delExposureAdjust(self):
        if "ExposureAdjust" in self.configuration:
            del self.configuration["ExposureAdjust"]
            return self.reconfigure(True, "ExposureAdjust")
        return True

    def set_chromatic_aberration(self, r=1.07, g=1.05, b=1.03):

        self.load_filter(
            "chromatic aberration",
            f"""
        float r = tex2D(k_txcolor, l_texcoord_color.xy / chromatic_offset_r).r;
        float g = tex2D(k_txcolor, l_texcoord_color.xy / chromatic_offset_g).g;
        float b = tex2D(k_txcolor, l_texcoord_color.xy / chromatic_offset_b).b;
        o_color = float4(r,g,b, o_color.a);""",
            uniforms=[
                "float chromatic_offset_r",
                "float chromatic_offset_g",
                "float chromatic_offset_b"
            ],
            shader_inputs={"chromatic_offset_r": r, "chromatic_offset_g": g, "chromatic_offset_b": b}
        )

    def del_chromatic_aberration(self):
        self.remove_filter("chromatic aberration", ["chromatic_offset_r",
                                                    "chromatic_offset_g",
                                                    "chromatic_offset_b"], ["float2 chromatic_offset_g",
                                                                            "float2 chromatic_offset_g",
                                                                            "float2 chromatic_offset_b"])

    def set_vignette(self, radius, vignette_strength=0.2, order=15):
        self.load_filter("Vignette", """ 
        float vignette_amount = length(l_fragpos - 0.5) - (1 - vignette_radius);
        o_color.rgb *= 1.0 - smoothstep(0.0, 1 - vignette_strength, vignette_amount );
""",
                         uniforms=["float vignette_radius", "float vignette_strength"],
                         shader_inputs={"vignette_radius": radius, "vignette_strength": vignette_strength},
                         order=order)

    #snake_case alias:
    set_msaa = setMSAA
    del_msaa = delMSAA
    del_cartoon_ink = delCartoonInk
    set_half_pixel_shift = setHalfPixelShift
    del_half_pixel_shift = delHalfPixelShift
    set_inverted = setInverted
    del_inverted = delInverted
    del_view_glow = delViewGlow
    set_volumetric_lighting = setVolumetricLighting
    set_bloom = setBloom
    set_view_glow = setViewGlow
    set_ambient_occlusion = setAmbientOcclusion
    set_cartoon_ink = setCartoonInk
    del_bloom = delBloom
    del_ambient_occlusion = delAmbientOcclusion
    set_blur_sharpen = setBlurSharpen
    del_blur_sharpen = delBlurSharpen
    del_volumetric_lighting = delVolumetricLighting
    set_gamma_adjust = setGammaAdjust
    del_gamma_adjust = delGammaAdjust
    set_srgb_encode = setSrgbEncode
    del_srgb_encode = delSrgbEncode
    set_exposure_adjust = setExposureAdjust
    del_exposure_adjust = delExposureAdjust
    set_high_dynamic_range = setHighDynamicRange
    del_high_dynamic_range = delHighDynamicRange


if __name__ == "__main__":
    from direct.showbase.ShowBase import ShowBase

    s = ShowBase()
    p = s.loader.load_model("panda")
    p.set_pos((0, 100, 0))
    p.reparent_to(s.render)
    c = CommonFilters(s.win, s.cam)

    #c.add_uniform("float raise_amount")
    #c.add_shader_inputs({"raise_amount": 0.1})
    #c.add_filter("raise amount", "o_color += 0.5 * raise_amount;", 0)

    #c.add_uniform("float increase_red")
    #c.add_shader_input("increase_red", 0.1)
    #c.add_filter("increase red", "o_color.r += increase_red;", 0)

    #c.setSrgbEncode()
    #c.set_high_dynamic_range()

    # c.setMSAA(8)
    # c.delMSAA()

    #c.set_chromatic_aberration()
    c.set_bloom()

    s.accept("d", c.del_bloom)

    #c.setBlurSharpen(0.5)

    #c.set_gamma_adjust(1.4)
    #c.set_vignette(0.4, 0.6, order=15)
    #c.set_cartoon_ink()

    for count, line in enumerate(c.current_text.split("\n")):
        print(count, line)

    s.accept("c", c.del_chromatic_aberration)
    s.run()
