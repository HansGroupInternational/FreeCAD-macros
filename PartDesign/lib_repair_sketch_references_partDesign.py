#
# Code based on Assembly2 update part code
# many Assembly2 functions have been copied here, instead of imported, as to allow users to use this macro without having Assembly2 installed
#

import FreeCAD
import Part #from FreeCAD
import numpy, datetime
from numpy import sin, cos, tan, arctan2, arccos, arcsin, dot
norm = numpy.linalg.norm


def debugPrint( level, msg, timePrefix=False ):
    if level <= debugPrint.printLevel:
        tPrefix = '' if not timePrefix else datetime.datetime.now().strftime('%H:%M:%S ')
        FreeCAD.Console.PrintMessage( '%s%s\n' % (tPrefix,msg))
debugPrint.printLevel = 2


class SketchReference:
    label = 'SketchReference'
    def __init__(self, object, subelement):
        self.object = object
        self.description = SketchReferenceInfo( subelement, object )
        self.DescriptionClass = SketchReferenceInfo

    def getShapeElementName( self ):
        se_org = self.description
        se_org_name = se_org.name
        T = ReversePlacementTransformWithBoundsNormalization( self.object )
        try:
            se_current = self.DescriptionClass( se_org_name, self.object, T )
            if se_org == se_current:
                return se_org_name
        except (IndexError, ValueError, RuntimeError):
            pass
        debugPrint(3, '%s %s.%s has changed. Searching for closest match' % ( self.label, self.object.Name, se_org_name) )

        prefixDict = {'Vertexes':'Vertex','Edges':'Edge','Faces':'Face'}
        if se_org_name.startswith('Vertex'):
            listName = 'Vertexes'
        elif se_org_name.startswith( 'Edge' ):
            listName = 'Edges'
        elif se_org_name.startswith( 'Face' ):
            listName = 'Faces'
        se_errors = []
        for j, subelement in enumerate( getattr( self.object.Shape, listName) ):
            se_name =  '%s%i' % (prefixDict[listName], j+1 )
            try:
                se_category = classifyShapeElement( self.object, se_name )
            except RuntimeError:
                FreeCAD.Console.PrintError('failure to classify %s.%s\n' % ( self.object.Name, se_name ) )
                continue
            if se_category == se_org.category:
                se_errors.append( self.DescriptionClass( se_name, self.object, T) - se_org )
        if debugPrint.printLevel >= 4:
            for e in se_errors:
                debugPrint(4, '    %s' % str(e))
        min_error = min( se_errors )
        return min_error.se1.name

class SketchReferenceInfo:
    def __init__(self, elementName, obj, T=None):
        self.name = elementName
        if T == None:
            T =  ReversePlacementTransformWithBoundsNormalization( obj )
        self.category = classifyShapeElement( obj, elementName )
        #if self.category in ['cylindricalSurface','circularEdge','plane','linearEdge']:
        #    self.axis_T = T.unRotate( getElementAxis( obj, shapeElementName ) ) #not important for Sketch References
        self.pos_T = T( getElementPos( obj, elementName ) )
        if not elementName.startswith('Vertex'):
            ind = int( elementName[4:] ) -1 
            element = obj.Shape.Faces[ind]  if elementName.startswith('Face') else obj.Shape.Edges[ind]
            self.vertex_P =  numpy.array( [ T(v.Point) for v in element.Vertexes ] )

    def _vertexesDifference( self, b):
        sqr_dist = 0
        for i in range( self.vertex_P.shape[0] ): #no. of rows
            i_array = numpy.repeat( self.vertex_P[i:i+1], b.vertex_P.shape[0], axis=0 )
            sqr_diff = (b.vertex_P - i_array) ** 2
            sqr_dist = sqr_dist + min( sqr_diff.sum(axis=1) )
        return sqr_dist

    def cmpErrors( self, b):
        if self.category in ['circularEdge','cylindricalSurface']:
            error1 = norm( self.pos_T - b.pos_T )
            error2 = self._vertexesDifference( b )
        elif self.category in ['plane', 'linearEdge']:
            error1 = 0
            error2 = self._vertexesDifference( b )
        elif self.category == 'vertex':
            error1 = 0
            error2 = norm( self.pos_T - b.pos_T )
        else:
            raise NotImplementedError, "logic not programmed for category %s" % self.category
        return error1, error2

    def __eq__( self, b, tol=10**-9 ):
        if self.category == b.category:
            error1, error2 = self.cmpErrors( b ) 
            return error1 < tol and error2 < tol
        else:
            return False

    def __sub__(self, b):
        return ShapeElement_Absolute_Difference( self, b )


class ShapeElement_Absolute_Difference:
    def __init__(self, shapeElement1, shapeElement2):
        self.se1 = shapeElement1
        self.se2 =  shapeElement2
        self.error1, self.error2 = shapeElement1.cmpErrors( shapeElement2 )
    def __lt__( self, b, tol=10**-9):
        if abs(self.error1 - b.error1) > tol:
            return self.error1 < b.error1
        else:
            return self.error2 < b.error2
    def __str__( self ):
        return '<diff %s-%s, errors  %1.2e  %1.2e>' % ( self.se1.name, self.se2.name, self.error1, self.error2 )


class SketchSupportReference(SketchReference):
    label = 'SketchSupportReference'
    def __init__(self, sketch):
        object, elementName = sketch.Support[0], sketch.Support[1][0]
        self.object = object
        self.DescriptionClass = SketchSupportReferenceInfo
        try:
            self.description = SketchSupportReferenceInfo( elementName , object )
        except IndexError: #not existent support ...
            self.description = SketchSupportReferenceInfo_nonExistentSupport( obj, sketch)


class SketchSupportReferenceInfo(SketchReferenceInfo):
    def __init__(self, elementName, obj, T=None):
        self.name = elementName
        if T == None:
            T =  ReversePlacementTransformWithBoundsNormalization( obj )
        self.category = classifyShapeElement( obj, elementName )
        self.axis_T = T.unRotate( getElementAxis( obj, elementName ) ) #not important for Sketch References
        self.pos_T = T( getElementPos( obj, elementName ) )
    def cmpErrors( self, b):
        error1 = 1 - abs( dot( self.axis_T, b.axis_T ) )
        error2 = norm( self.pos_T - b.pos_T )
        return error1, error2

class SketchSupportReferenceInfo_nonExistentSupport(SketchSupportReferenceInfo):
    def __init__(self, obj, sketch ):
        self.name = 'Face_Unparasable_Sketch_Support'
        T =  ReversePlacementTransformWithBoundsNormalization( obj )
        self.category = 'plane'
        self.axis_T = T.unRotate( sketch.Shape.Placement.Rotation.Axis ) 
        self.pos_T = T( sketch.Shape.Placement.Base )

class DirectionReference(SketchReference):
    label = 'DirectionReference'
    def __init__(self, object, elementName):
        self.object = object
        self.DescriptionClass = DirectionReferenceInfo
        self.description = self.DescriptionClass( elementName , object )

class DirectionReferenceInfo(SketchSupportReferenceInfo):
    def cmpErrors( self, b):
        error1 = 1 - dot( self.axis_T, b.axis_T )
        error2 = norm( self.pos_T - b.pos_T )
        return error1, error2


def classifyShapeElement( obj, shapeElementName ):
    if shapeElementName.startswith('Face'):
        ind = int( shapeElementName[4:] ) -1 
        face = obj.Shape.Faces[ind]
        if str(face.Surface) == '<Plane object>':
            return 'plane'
        elif hasattr(face.Surface,'Radius') or str(face.Surface).startswith('<SurfaceOfRevolution'):
            return 'cylindricalSurface'
        elif str( face.Surface ).startswith('Sphere '):
            return 'sphericalSurface'
    elif shapeElementName.startswith('Edge'):
        ind = int( shapeElementName[4:]) -1 
        edge = obj.Shape.Edges[ind]
        if hasattr( edge, 'Curve'): #issue 39 assembly2
            if isinstance(edge.Curve, Part.Line):
                return 'linearEdge'
            elif hasattr( edge.Curve, 'Radius' ):
                return 'circularEdge'
            BSpline = edge.Curve.toBSpline()
            arcs = BSpline.toBiArcs(10**-6)
            if all( hasattr(a,'Center') for a in arcs ):
                centers = numpy.array([a.Center for a in arcs])
                sigma = numpy.std( centers, axis=0 )
                if max(sigma) < 10**-6: #then circular curce
                    return 'circularEdge'
            if all(isinstance(a, Part.Line) for a in arcs):
                lines = arcs
                D = numpy.array([L.tangent(0)[0] for L in lines]) #D(irections)
                if numpy.std( D, axis=0 ).max() < 10**-9: #then linear curve
                    return 'linearEdge'
    elif shapeElementName.startswith('Vertex'):
        return 'vertex'
    raise RuntimeError, "unable to classify subelement %s.%s" % (obj.Name, shapeElementName)




#assembly2lib
def getElementPos(obj, elementName):
    pos = None
    if elementName.startswith('Face'):
        ind = int( elementName[4:] ) -1 
        face = obj.Shape.Faces[ind]
        surface = face.Surface
        if str(surface) == '<Plane object>':
            pos = surface.Position
        elif all( hasattr(surface,a) for a in ['Axis','Center','Radius'] ):
            pos = surface.Center
        elif str(surface).startswith('<SurfaceOfRevolution'):
            pos = face.Edges[0].Curve.Center
        else: #numerically approximating surface
            plane_norm, plane_pos, error = fit_plane_to_surface1(face.Surface)
            error_normalized = error / face.BoundBox.DiagonalLength
            if error_normalized < 10**-6: #then good plane fit
                pos = plane_pos
            axis, center, error = fit_rotation_axis_to_surface1(face.Surface)
            error_normalized = error / face.BoundBox.DiagonalLength
            if error_normalized < 10**-6: #then good rotation_axis fix
                pos = center
    elif elementName.startswith('Edge'):
        ind = int( elementName[4:]) -1 
        edge = obj.Shape.Edges[ind]
        if isinstance(edge.Curve, Part.Line):
            pos = edge.Curve.StartPoint
        elif hasattr( edge.Curve, 'Center'): #circular curve
            pos = edge.Curve.Center
        else:
            BSpline = edge.Curve.toBSpline()
            arcs = BSpline.toBiArcs(10**-6)
            if all( hasattr(a,'Center') for a in arcs ):
                centers = numpy.array([a.Center for a in arcs])
                sigma = numpy.std( centers, axis=0 )
                if max(sigma) < 10**-6: #then circular curce
                    pos = centers[0]
            if all(isinstance(a, Part.Line) for a in arcs):
                lines = arcs
                D = numpy.array([L.tangent(0)[0] for L in lines]) #D(irections)
                if numpy.std( D, axis=0 ).max() < 10**-9: #then linear curve
                    return lines[0].value(0)
    elif elementName.startswith('Vertex'):
        ind = int( elementName[6:]) -1 
        vertex = obj.Shape.Vertexes[ind]
        return  vertex.Point
    if pos <> None:
        return numpy.array(pos)
    else:
        raise NotImplementedError,"getElementPos Failed! Locals:\n%s" % formatDictionary(locals(),' '*4)

#assembly2lib
def getElementAxis(obj, elementName):
    axis = None
    if elementName.startswith('Face'):
        ind = int( elementName[4:] ) -1 
        face = obj.Shape.Faces[ind]
        surface = face.Surface
        if hasattr(surface,'Axis'):
            axis = surface.Axis
        elif str(surface).startswith('<SurfaceOfRevolution'):
            axis = face.Edges[0].Curve.Axis
        else: #numerically approximating surface
            plane_norm, plane_pos, error = fit_plane_to_surface1(face.Surface)
            error_normalized = error / face.BoundBox.DiagonalLength
            if error_normalized < 10**-6: #then good plane fit
                axis = plane_norm
            axis_fitted, center, error = fit_rotation_axis_to_surface1(face.Surface)
            error_normalized = error / face.BoundBox.DiagonalLength
            if error_normalized < 10**-6: #then good rotation_axis fix
                axis = axis_fitted
    elif elementName.startswith('Edge'):
        ind = int( elementName[4:]) -1 
        edge = obj.Shape.Edges[ind]
        if isinstance(edge.Curve, Part.Line):
            axis = edge.Curve.tangent(0)[0]
        elif hasattr( edge.Curve, 'Axis'): #circular curve
            axis =  edge.Curve.Axis
        else:
            BSpline = edge.Curve.toBSpline()
            arcs = BSpline.toBiArcs(10**-6)
            if all( hasattr(a,'Center') for a in arcs ):
                centers = numpy.array([a.Center for a in arcs])
                sigma = numpy.std( centers, axis=0 )
                if max(sigma) < 10**-6: #then circular curce
                    axis = a.Axis
            if all(isinstance(a, Part.Line) for a in arcs):
                lines = arcs
                D = numpy.array([L.tangent(0)[0] for L in lines]) #D(irections)
                if numpy.std( D, axis=0 ).max() < 10**-9: #then linear curve
                    return D[0]
    if axis <> None:
        return numpy.array(axis)
    else:
        raise NotImplementedError,"getElementAxis Failed! Locals:\n%s" % formatDictionary(locals(),' '*4)


# assembly2.variableManager
class ReversePlacementTransformWithBoundsNormalization:
    def __init__(self, obj):
        x, y, z = obj.Placement.Base.x, obj.Placement.Base.y, obj.Placement.Base.z
        self.offset = numpy.array([x, y, z]) #placement offset
        axis, theta = quaternion_to_axis_and_angle( *obj.Placement.Rotation.Q )
        if theta <> 0:
            azi, ela = axis_to_azimuth_and_elevation_angles(*axis)
        else:
            azi, ela = 0, 0
        self.R = azimuth_elevation_rotation_matrix( azi, ela, theta ) #placement rotation
        #now for bounds normalization
        #V = [ self.undoPlacement(v.Point) for v in obj.Shape.Vertexes] #no nessary in BoundBox is now used.
        V = [] 
        BB = obj.Shape.BoundBox
        extraPoints = []
        for z in [ BB.ZMin, BB.ZMax ]:
            for y in [ BB.YMin, BB.YMax ]:
                for x in [ BB.XMin, BB.XMax ] :
                    V.append( self.undoPlacement([x,y,z]) )
        V = numpy.array(V)
        self.Bmin = V.min(axis=0)
        self.Bmax = V.max(axis=0)
        self.dB = self.Bmax - self.Bmin

    def undoPlacement(self, p):
        # p = R*q + offset
        return numpy.linalg.solve( self.R, numpy.array(p) - self.offset )
    
    def unRotate(self, p):
        return numpy.linalg.solve( self.R, p)

    def __call__( self, p):
        q = self.undoPlacement(p)
        # q = self.Bmin + r* self.dB (where r is in normilezed coordinates)
        return (q - self.Bmin) / self.dB
    
# directly copied from assembly2.lib3D for all the rest

def arcsin2( v, allowableNumericalError=10**-1 ):
    if -1 <= v and v <= 1:
        return arcsin(v)
    elif abs(v) -1 < allowableNumericalError:
        return pi/2 if v > 0 else -pi/2
    else:
        raise ValueError,"arcsin2 called with invalid input of %s" % v

def quaternion_to_axis_and_angle(  q_1, q_2, q_3, q_0): 
    'http://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions'
    q =  numpy.array( [q_1, q_2, q_3])
    if norm(q) > 0:
        return q/norm(q), 2*arccos2(q_0)
    else:
        return numpy.array([1.0,0,0]), 2*arccos2(q_0)

def arccos2( v, allowableNumericalError=10**-1 ):
    if -1 <= v and v <= 1:
        return arccos(v)
    elif abs(v) -1 < allowableNumericalError:
        return 0 if v > 0 else pi
    else:
        raise ValueError,"arccos2 called with invalid input of %s" % v

def normalize( v ):
    return v / norm(v)

def azimuth_and_elevation_angles_to_axis( a, e):
    u_z = sin(e)
    u_x = cos(e)*cos(a)
    u_y = cos(e)*sin(a)
    return numpy.array([ u_x, u_y, u_z ])

def axis_to_azimuth_and_elevation_angles( u_x, u_y, u_z ):
    return arctan2( u_y, u_x), arcsin2(u_z)

def axis_rotation_matrix( theta, u_x, u_y, u_z ):
    ''' http://en.wikipedia.org/wiki/Rotation_matrix '''
    return numpy.array( [
            [ cos(theta) + u_x**2 * ( 1 - cos(theta)) , u_x*u_y*(1-cos(theta)) - u_z*sin(theta) ,  u_x*u_z*(1-cos(theta)) + u_y*sin(theta) ] ,
            [ u_y*u_x*(1-cos(theta)) + u_z*sin(theta) , cos(theta) + u_y**2 * (1-cos(theta))    ,  u_y*u_z*(1-cos(theta)) - u_x*sin(theta )] ,
            [ u_z*u_x*(1-cos(theta)) - u_y*sin(theta) , u_z*u_y*(1-cos(theta)) + u_x*sin(theta) ,              cos(theta) + u_z**2*(1-cos(theta))   ]
            ])

def axis_rotation( p, theta, u_x, u_y, u_z ):
    return dotProduct(axis_rotation_matrix( theta, u_x, u_y, u_z ), p)

def azimuth_elevation_rotation_matrix(azi, ela, theta ):
    #print('azimuth_and_elevation_angles_to_axis(azi, ela) %s' % azimuth_and_elevation_angles_to_axis(azi, ela))
    return axis_rotation_matrix( theta, *azimuth_and_elevation_angles_to_axis(azi, ela))
