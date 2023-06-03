import time
import os
import sys
sys.path.append('/home/xianghui_yang/anaconda3/envs/cudatoolkit-11.1/bin/python')
print(sys.path)
print(sys.exec_prefix)
import contextlib
from math import radians
# from PIL import Image
import random
import bpy
import json

SHAPENET_ROOT = '/home/xianghui_yang/data/ShapeNet/ShapeNetCore.v1/'
DIR_RENDERING_PATH = '/home/xianghui_yang/data/ShapeNet/my_ShapeNetRendering_depth_offset'
TEST_RENDERING_PATH = '/home2/xieyunwei/occupancy_networks/data/render_test'
RENDERING_MAX_CAMERA_DIST = 1.75
N_VIEWS = 3
RENDERING_BLENDER_TMP_DIR = '/tmp/blender'

def voxel2mesh(voxels):
    cube_verts = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]]  # 8 points

    cube_faces = [[0, 1, 2],
                  [1, 3, 2],
                  [2, 3, 6],
                  [3, 7, 6],
                  [0, 2, 6],
                  [0, 6, 4],
                  [0, 5, 1],
                  [0, 4, 5],
                  [6, 7, 5],
                  [6, 5, 4],
                  [1, 7, 3],
                  [1, 5, 7]]  # 12 face

    cube_verts = np.array(cube_verts)
    cube_faces = np.array(cube_faces) + 1

    l, m, n = voxels.shape

    scale = 0.01
    cube_dist_scale = 1.1
    verts = []
    faces = []
    curr_vert = 0
    for i in range(l):
        for j in range(m):
            for k in range(n):
                # If there is a non-empty voxel
                if voxels[i, j, k] > 0:
                    verts.extend(scale * (cube_verts + cube_dist_scale * np.array([[i, j, k]])))
                    faces.extend(cube_faces + curr_vert)
                    curr_vert += len(cube_verts)

    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


class BaseRenderer:
    model_idx   = 0

    def __init__(self):
        # bpy.data.scenes['Scene'].render.engine = 'CYCLES'
        # bpy.context.scene.cycles.device = 'GPU'
        # bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        # bpy.context.user_preferences.system.compute_device = 'CUDA_1'

        # changing these values does affect the render.

        # remove the default cube
        bpy.ops.object.select_pattern(pattern="Cube")
        bpy.ops.object.delete()

        render_context = bpy.context.scene.render
        world  = bpy.context.scene.world
        camera = bpy.data.objects['Camera']
        light_1  = bpy.data.objects['Lamp']
        light_1.data.type = 'HEMI'

        # set the camera postion and orientation so that it is in
        # the front of the object
        camera.location       = (1, 0, 0)
        camera.rotation_mode  = 'ZXY'
        camera.rotation_euler = (0, radians(90), radians(90))

        # parent camera with a empty object at origin
        org_obj                = bpy.data.objects.new("RotCenter", None)
        org_obj.location       = (0, 0, 0)
        org_obj.rotation_euler = (0, 0, 0)
        bpy.context.scene.objects.link(org_obj)

        camera.parent = org_obj  # setup parenting

        # render setting
        render_context.resolution_percentage = 100
        world.horizon_color = (1, 1, 1)  # set background color to be white

        # set file name for storing rendering result
        self.result_fn = '%s/render_result_%d.png' % (DIR_RENDERING_PATH, os.getpid())
        bpy.context.scene.render.filepath = self.result_fn

        # new settings
        # bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        # bpy.context.scene.render.image_settings.file_format = 'PNG'
        # bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        # bpy.context.scene.render.image_settings.use_zbuffer = True
        
        self.render_context = render_context
        self.org_obj = org_obj
        self.camera = camera
        self.light = light_1
        self._set_lighting()

    def initialize(self, models_fn, viewport_size_x, viewport_size_y):
        self.models_fn = models_fn
        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y

    def _set_lighting(self):
        pass

    def setViewpoint(self, azimuth, altitude, yaw, distance_ratio, fov):
        self.org_obj.rotation_euler = (0, 0, 0)
        self.light.location = (distance_ratio *
                               (RENDERING_MAX_CAMERA_DIST + 2), 0, 0)
        self.camera.location = (distance_ratio *
                                RENDERING_MAX_CAMERA_DIST, 0, 0)
        self.org_obj.rotation_euler = (radians(-yaw),
                                       radians(-altitude),
                                       radians(-azimuth))

    def setTransparency(self, transparency):
        """ transparency is either 'SKY', 'TRANSPARENT'
        If set 'SKY', render background using sky color."""
        self.render_context.alpha_mode = transparency

    def selectModel(self):
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_pattern(pattern="RotCenter")
        bpy.ops.object.select_pattern(pattern="Lamp*")
        bpy.ops.object.select_pattern(pattern="Camera")
        bpy.ops.object.select_all(action='INVERT')

    def printSelection(self):
        print(bpy.context.selected_objects)

    def clearModel(self):
        self.selectModel()
        bpy.ops.object.delete()

        # The meshes still present after delete
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        for item in bpy.data.materials:
            bpy.data.materials.remove(item)

    def setModelIndex(self, model_idx):
        self.model_idx = model_idx

    def loadModel(self, file_path=None):
        if file_path is None:
            file_path = self.models_fn[self.model_idx]

        if file_path.endswith('obj'):
            bpy.ops.import_scene.obj(filepath=file_path)
        elif file_path.endswith('3ds'):
            bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
        elif file_path.endswith('dae'):
            # Must install OpenCollada. Please read README.md
            bpy.ops.wm.collada_import(filepath=file_path)
        else:
            raise Exception("Loading failed: %s Model loading for type %s not Implemented" %
                            (file_path, file_path[-4:]))

    def render(self, load_model=True, clear_model=True, resize_ratio=None,
               return_image=True, image_path=os.path.join(RENDERING_BLENDER_TMP_DIR, 'tmp.png')):
        """ Render the object """
        if load_model:
            self.loadModel()

        # resize object
        self.selectModel()
        if resize_ratio:
            bpy.ops.transform.resize(value=resize_ratio)

        self.result_fn = image_path
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # save straight to file

        if resize_ratio:
            bpy.ops.transform.resize(value=(1/resize_ratio[0],
                1/resize_ratio[1], 1/resize_ratio[2]))

        if clear_model:
            self.clearModel()


class ShapeNetRenderer(BaseRenderer):

    def __init__(self):
        super().__init__()
        self.setTransparency('TRANSPARENT')

    def _set_lighting(self):
        # Create new lamp datablock
        light_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')

        # Create new object with our lamp datablock
        light_2 = bpy.data.objects.new(name="New Lamp", object_data=light_data)
        bpy.context.scene.objects.link(light_2)

        # put the light behind the camera. Reduce specular lighting
        self.light.location       = (0, -2, 2)
        self.light.rotation_mode  = 'ZXY'
        self.light.rotation_euler = (radians(45), 0, radians(90))
        self.light.data.energy = 0.7

        light_2.location       = (0, 2, 2)
        light_2.rotation_mode  = 'ZXY'
        light_2.rotation_euler = (-radians(45), 0, radians(90))
        light_2.data.energy = 0.7


class VoxelRenderer(BaseRenderer):

    def __init__(self):
        super().__init__()
        self.setTransparency('SKY')

    def _set_lighting(self):
        self.light.location       = (0, 3, 3)
        self.light.rotation_mode  = 'ZXY'
        self.light.rotation_euler = (-radians(45), 0, radians(90))
        self.light.data.energy = 0.7

        # Create new lamp datablock
        light_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')

        # Create new object with our lamp datablock
        light_2 = bpy.data.objects.new(name="New Lamp", object_data=light_data)
        bpy.context.scene.objects.link(light_2)

        light_2.location       = (4, 1, 6)
        light_2.rotation_mode  = 'XYZ'
        light_2.rotation_euler = (radians(37), radians(3), radians(106))
        light_2.data.energy = 0.7

    def render_voxel(self, pred, thresh=0.4,
                     image_path=os.path.join(RENDERING_BLENDER_TMP_DIR, 'tmp.png')):
        # Cleanup the scene
        self.clearModel()
        out_f = os.path.join(RENDERING_BLENDER_TMP_DIR, 'tmp.obj')
        occupancy = pred > thresh
        vertices, faces = voxel2mesh(occupancy)
        with contextlib.suppress(IOError):
            os.remove(out_f)
        write_obj(out_f, vertices, faces)

        # Load the obj
        bpy.ops.import_scene.obj(filepath=out_f)
        bpy.context.scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True)  # save straight to file


import argparse
def main(args):
    if not os.path.exists(os.path.join(DIR_RENDERING_PATH, args.model_class)):
        print('Create', os.path.join(DIR_RENDERING_PATH, args.model_class))
        os.mkdir(os.path.join(DIR_RENDERING_PATH, args.model_class))
    file_paths = []
    # all_model_class = []
    all_model_ids = []

    # with open(args.task_file, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if line == '': 
    #             continue

    #         tmp = line.rstrip('\n').split(' ')
    #         all_model_class.append(tmp[0])
    #         all_model_ids.append(tmp[1])
    #         file_paths.append(os.path.join(SHAPENET_ROOT, tmp[0], tmp[1], 'model.obj'))

    # all shapenet
    # for fn in os.listdir(os.path.join(SHAPENET_ROOT, args.model_class)):
    #     file_paths.append(os.path.join(SHAPENET_ROOT, args.model_class, fn, 'model.obj'))
    #     all_model_ids.append(fn)

    # all 0shot shapenet test
    with open('/home/xianghui_yang/data/ShapeNet/split.json', 'r') as file:
        data_split = json.load(file)
    for sid in data_split["test"]:
        for mid in data_split["test"][sid]:
            file_paths.append(os.path.join(SHAPENET_ROOT, sid, mid, 'model.obj'))
            all_model_ids.append((sid, mid))

    print('e.g.',file_paths[:5])

    sum_time = 0
    renderer = ShapeNetRenderer()
    renderer.initialize(file_paths, 137, 137)
    for i, curr_model_id in enumerate(all_model_ids):
        start = time.time()
        # rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, args.model_class, all_model_ids[i])
        sid, mid = curr_model_id
        rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, sid, mid)

        if not os.path.exists(rendering_curr_model_root):
            os.makedirs(rendering_curr_model_root)
            os.mkdir(os.path.join(rendering_curr_model_root, 'rendering'))

        if os.path.exists(os.path.join(rendering_curr_model_root, 'rendering', '%02d.png' % (N_VIEWS - 1))):
            continue

        with open( os.path.join(rendering_curr_model_root, 'rendering', 'renderings.txt'), 'w' ) as f:
            for view_id in range(N_VIEWS):
                print('%.2d' % view_id, file = f)

        camera_file_f = open(os.path.join(rendering_curr_model_root, 'rendering', 'rendering_metadata.txt'), 'w')
        
        # raw
        # for view_id in range(N_VIEWS):
        #     image_path = os.path.join(rendering_curr_model_root, 'rendering', '%02d.png' % view_id)

        #     az, el, depth_ratio = [360 * random.random(), 5 * random.random() + 25, 0.3 * random.random() + 0.65]
        
        # change depth 1% 5% 10%
        with open(os.path.join('/home/xianghui_yang/data/ShapeNet/ShapeNetRendering', sid, mid, 'rendering/rendering_metadata.txt'),'r') as f:
            param_list = f.readlines()
            param = param_list[0].rstrip() 
            param = param.split(' ')
        az = float(param[0])
        el = float(param[1])
        cur_depth = float(param[3])
        depth_ratio_list = [cur_depth*1.01, cur_depth*1.05, cur_depth*1.1]

        for view_id in range(N_VIEWS):
            image_path = os.path.join(rendering_curr_model_root, 'rendering', '%02d.png' % view_id)

            depth_ratio = depth_ratio_list[view_id]

            renderer.setModelIndex(i)
            renderer.setViewpoint(az, el, 0, depth_ratio, 25)

            if view_id == 0:
                load_model_flag = True
            else:
                load_model_flag = False

            if view_id == N_VIEWS - 1:
                clear_model_flag = True
            else:
                clear_model_flag = False

            renderer.render(load_model=load_model_flag, return_image=False,
                    clear_model=clear_model_flag, image_path=image_path)

            print(az, el, 0, depth_ratio, 25, file=camera_file_f)
            print('Saved at %s' % image_path)

        camera_file_f.close()

        end = time.time()
        sum_time += end - start

        if i % 10 == 0:
            print(sum_time/(10))
            sum_time = 0

def main_single(args):
    file_paths = [os.path.join(SHAPENET_ROOT, args.model_class, args.model_id, 'model.obj')]
    renderer = ShapeNetRenderer()
    renderer.initialize(file_paths, 137, 137)

    rendering_curr_model_root = os.path.join(DIR_RENDERING_PATH, args.model_class, args.model_id)

    if not os.path.exists(rendering_curr_model_root):
        os.mkdir(rendering_curr_model_root)

    if os.path.exists(os.path.join(rendering_curr_model_root, 'rendering_exr', '%.2d.exr' % (N_VIEWS - 1))):
        return

    with open( os.path.join(rendering_curr_model_root, 'renderings.txt'), 'w' ) as f:
        for view_id in range(N_VIEWS):
            print('%.2d' % view_id, file = f)

    camera_file_f = open(os.path.join(rendering_curr_model_root, 'rendering_metadata.txt'), 'w')
    
    for view_id in range(N_VIEWS):
        image_path = os.path.join(rendering_curr_model_root, 'rendering_exr', '%.2d.exr' % view_id)

        az, el, depth_ratio = [360 * random.random(), 5 * random.random() + 25, 0.3 * random.random() + 0.65]
    
        renderer.setModelIndex(0)
        renderer.setViewpoint(az, el, 0, depth_ratio, 25)

        if view_id == 0:
            load_model_flag = True
        else:
            load_model_flag = False

        if view_id == N_VIEWS - 1:
            clear_model_flag = True
        else:
            clear_model_flag = False

        renderer.render(load_model=load_model_flag, return_image=False,
                clear_model=clear_model_flag, image_path=image_path)

        print(az, el, 0, depth_ratio, 25, file=camera_file_f)
        print('Saved at %s' % image_path)

    camera_file_f.close()

def test():
    """Test function"""
    # Modify the following file to visualize the model
    #dn = '/home2/xieyunwei/occupancy_networks/external/ShapeNetCore.v1/02958343/'
    #model_id = ['2c981b96364b1baa21a66e8dfcce514a']
    dn = '/home2/xieyunwei/occupancy_networks/external/ShapeNetCore.v1/02958343/'
    model_id = ['f9c1d7748c15499c6f2bd1c4e9adb41']
    file_paths = [os.path.join(dn, m_id, 'model.obj') for m_id in model_id]
    sum_time = 0
    renderer = ShapeNetRenderer()
    renderer.initialize(file_paths, 137, 137)
    for i, curr_model_id in enumerate(model_id):
        start = time.time()
        image_path = '%s/%s.exr' % (TEST_RENDERING_PATH, curr_model_id)

        az, el, depth_ratio = [360 * random.random(), 5 * random.random() + 25, 0.3 * random.random() + 0.65]
    
        renderer.setModelIndex(i)
        renderer.setViewpoint(30, 30, 0, 0.7, 25)

        renderer.render(load_model=True, return_image=False,
                clear_model=True, image_path=image_path)

        print('Saved at %s' % image_path)

        end = time.time()
        sum_time += end - start
        if i % 10 == 0:
            print(sum_time/(10))
            sum_time = 0

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render according to a task_split_file')
    parser.add_argument('--task_file', type=str, default='', help='task split file')
    parser.add_argument('--single', action='store_true', help='use single')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--model_class', type=str, default='', help='model class')
    parser.add_argument('--model_id', type=str, default='', help='model id')
    print(sys.argv)
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)
    # args = parser.parse_args()
    print(args)
    if args.test:
        test()
        exit(0)

    if not args.single:
        main(args)
    else:
        main_single(args)