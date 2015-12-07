
import lib_repair_sketch_references_partDesign
from lib_repair_sketch_references_partDesign import *
from PySide import QtGui, QtCore


debugPrint(2,'repair_sketch_references started',timePrefix=True)

reference_state = lib_repair_sketch_references_partDesign.reference_state
if len( reference_state ) == 0:
    raise ValueError, "record_sketch_references needs to be run first"

FreeCAD.ActiveDocument.openTransaction('repair_references') #undo

progress_n = len( reference_state )
progress_ind = 0
progressDialog = QtGui.QProgressDialog( "Repair Progress", "Cancel", 0, progress_n )
progressDialog.setWindowModality(QtCore.Qt.WindowModal)
progressDialog.forceShow()


for obj_name, exGeom, supportGeom in reference_state:
    if progressDialog.wasCanceled():
        break
    progressDialog.setLabelText(obj_name)
    progressDialog.setValue( progress_ind )
    progress_ind = progress_ind + 1

    debugPrint(3,'  checking %s' % obj_name)
    obj = FreeCAD.ActiveDocument.getObject( obj_name )
    if obj.TypeId == 'Sketcher::SketchObject':
        sketch_name = obj_name
        sketch = obj
        sketch = FreeCAD.ActiveDocument.getObject( sketch_name )
        assert sketch != None
        changed = False

        newExtGeom = []
        for g in exGeom:
            se_name = g.getShapeElementName()
            old_name = g.description.name
            if se_name != old_name:
                changed = True
                debugPrint( 2,'  %s.ExternalGeometry:  %s -> %s' % (sketch_name, old_name, se_name) )
            newExtGeom.append( ( g.object, se_name) )
        if len( newExtGeom ) != len( sketch.ExternalGeometry ) : # then deletion of unparasable external geometry
            debugPrint( 2,'  %s.ExternalGeometry trimmed to remove unparasable external geometry' % sketch_name )
            changed = True

        if supportGeom != None:
            se_name = supportGeom.getShapeElementName()
            old_name = supportGeom.description.name
            if se_name != old_name:
                changed = True
                debugPrint( 2,'  %s.Support:  %s -> %s' % (sketch_name, old_name, se_name) )
                sketch.Support = ( supportGeom.object, [se_name] )

        if changed:
            sketch.ExternalGeometry = newExtGeom 
            FreeCAD.ActiveDocument.recompute()
            debugPrint(3,'  FreeCAD.ActiveDocument.recompute()')
            #some sketch does not update correctly the first time ...
            sketch.solve()
            FreeCAD.ActiveDocument.recompute()


    elif obj.TypeId == 'PartDesign::LinearPattern':
        se_name = supportGeom.getShapeElementName()
        old_name = supportGeom.description.name
        if se_name != old_name:
            debugPrint( 2,'  %s.Direction:  %s -> %s' % (obj_name, old_name, se_name) )
            obj.Direction = ( supportGeom.object, [se_name] )
            obj.touch()
            FreeCAD.ActiveDocument.recompute()

    elif obj.TypeId in ['PartDesign::Fillet', 'PartDesign::Chamfer']:
        changed = False
        newExtGeom = []
        for g in exGeom:
            se_name = g.getShapeElementName()
            old_name = g.description.name
            if se_name != old_name:
                changed = True
                refObj = g.object
                debugPrint( 2,'  %s.EdgeReference:  %s -> %s' % (sketch_name, old_name, se_name) )
            newExtGeom.append( se_name )
        if changed:
            obj.Base = ( refObj , newExtGeom )
            obj.touch()
            FreeCAD.ActiveDocument.recompute()


if not progressDialog.wasCanceled():
    progressDialog.setValue( progress_ind )
    debugPrint(1,'repair_sketch_references completed',timePrefix=True)
else:
    debugPrint(1,'repair_sketch_references aborted, press undo',timePrefix=True)

FreeCAD.ActiveDocument.commitTransaction() #undo
