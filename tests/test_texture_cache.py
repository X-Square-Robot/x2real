import importlib.util
from pathlib import Path
import os
import tempfile
import unittest
from pxr import Usd, UsdShade, Sdf

from manaenv.utils import texture_cache as m

class TextureCacheTest(unittest.TestCase):
 def test_preserves_source_and_repairs_corrupt_cache(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/'Scan.jpg';source.write_bytes(b'original pixels');os.utime(source,(0,0))
   cached=m.cache_epoch_texture(source,root/'cache')
   self.assertEqual(source.read_bytes(),cached.read_bytes());self.assertEqual(source.stat().st_mtime_ns,0);self.assertGreater(cached.stat().st_mtime_ns,0)
   cached.write_bytes(b'corrupt');self.assertEqual(m.cache_epoch_texture(source,root/'cache').read_bytes(),b'original pixels')
 def test_relative_reference_session_only_and_idempotent(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);asset=root/'asset';asset.mkdir();image=asset/'Scan.jpg';image.write_bytes(b'pixels');os.utime(image,(0,0))
   good=asset/'normal.png';good.write_bytes(b'normal')
   src=Usd.Stage.CreateNew(str(asset/'asset.usda'));p=src.DefinePrim('/World','Xform');src.SetDefaultPrim(p);shader=UsdShade.Shader.Define(src,'/World/Shader')
   shader.CreateInput('diffuse_texture',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('Scan.jpg'));shader.CreateInput('other_texture',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('normal.png'));src.GetRootLayer().Save()
   original_layer=(asset/'asset.usda').read_bytes()
   stage=Usd.Stage.CreateInMemory();stage.DefinePrim('/Object').GetReferences().AddReference(str(asset/'asset.usda'))
   self.assertEqual(m.repair_epoch_texture_resources(stage,root/'cache'),{'textures':1,'attributes':1})
   attr=stage.GetPrimAtPath('/Object/Shader').GetAttribute('inputs:diffuse_texture');self.assertEqual(Path(attr.Get().resolvedPath).read_bytes(),image.read_bytes())
   self.assertEqual(stage.GetPrimAtPath('/Object/Shader').GetAttribute('inputs:other_texture').Get().path,'normal.png')
   self.assertEqual(m.repair_epoch_texture_resources(stage,root/'cache'),{'textures':0,'attributes':0})
   self.assertEqual((asset/'asset.usda').read_bytes(),original_layer);self.assertEqual(image.stat().st_mtime_ns,0)
 def test_same_basename_different_pixels_never_share_cache(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);results=[]
   for name,data in [('a',b'first'),('b',b'second')]:
    parent=root/name;parent.mkdir();p=parent/'Scan.jpg';p.write_bytes(data);os.utime(p,(0,0));results.append(m.cache_epoch_texture(p,root/'cache'))
   self.assertNotEqual(*results)
if __name__=='__main__':unittest.main()
