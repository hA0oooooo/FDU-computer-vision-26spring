import bpy

def apply_vertex_color_material(obj):
    if obj.type != "MESH":
        return

    mesh = obj.data

    if len(mesh.color_attributes) == 0:
        print(f"{obj.name}: no color attributes, skip")
        return

    color_attr_name = "Col" if "Col" in mesh.color_attributes else mesh.color_attributes[0].name

    mat = bpy.data.materials.new(obj.name + "_vertex_color_mat")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in list(nodes):
        nodes.remove(node)

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (450, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (180, 0)

    attr = nodes.new(type="ShaderNodeAttribute")
    attr.location = (-120, 0)
    attr.attribute_name = color_attr_name

    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.85
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 1.0

    mesh.materials.clear()
    mesh.materials.append(mat)

    print(f"{obj.name}: use color attributes {color_attr_name}")

for obj in bpy.context.scene.objects:
    apply_vertex_color_material(obj)

for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = "MATERIAL"

print("finish")