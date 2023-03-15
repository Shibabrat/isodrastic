#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for functions in the toy tokamak model

"""

import numpy as np
import numpy.linalg as la

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# TODO: 1. Add the vector field for the first order GCM
# def first_order_eom():
#     """
#     Equations for the first-oder guiding center motion
#     """
# # TODO: 1. Add the vector field for the non-axisymmetric toy tokamak

#     return np.array([rDot, phiDot, zDot])

def intersect_poloidalplane(t, y, *field_parameters):
    """
    Compute intersection of fieldlines with a poloidal plane defined
    by phi = constant where phi is the toroidal coordinate.
    """
    phi = np.arctan2(y[1], y[0])
    return phi - np.pi/2
intersect_poloidalplane.terminal = False
intersect_poloidalplane.direction = 1


def points_poloidal_plane(resolution_Rz, range_Rz):
    """
    Returns (R, z) coordinates for a constant azimuthal angle
   
    Parameters
    ----------
    resolution: list of size 2
    Number of grid points in the R, z coordinates 
    
    range: list of size 4
    Minimum and maximum coordinates for generating the grid points 
    """
    
    min_zrange, max_zrange, min_Rrange, max_Rrange = range_Rz
    grid_R = np.linspace(min_Rrange, max_Rrange, resolution_Rz[0] + 1)
    grid_z = np.linspace(min_zrange, max_zrange, resolution_Rz[1] + 1)
    
    mesh_R, mesh_z = np.meshgrid(grid_R, grid_z)
    
    return mesh_R, mesh_z

def field_comps(points, field_parameters):
    """
    field_comps(points, field_parameters)
    
    Returns the field components in the form (n, 3) 
    where n is the number of points in R^3
    
    Parameters
    ----------
    points : ndarray of size (n, 3)
    Cylindrical polar coordinates of n points 
    
    field_parameters : ndarray of size (1, 2)
    C and R0 values of the toy tokamak model
    
    Returns
    -------
    out : ndarray of size (n, 3)
    Array of field components in column ordered as (B_R, B_{\phi}, B_z)
    evaluated at the n points
    
    """

    R0, C, epsilon, initialEnergy = field_parameters

    if np.size(points, 0) == 3:
        R, phi, z = points
    else:
        R = points[:, 0]
        phi = points[:, 1]
        z = points[:, 2]

    # radialComps = - points[:,2]/points[:,0] 
    # phiComps = C/points[:,0]
    # zComps = (points[:,0] - R0)/points[:,0] 

    # radialComps = - (z + epsilon*np.cos(phi))/R
    # phiComps = C/R
    # zComps = (R - R0)/R 

    radialComps = - (z / R + epsilon * np.cos(phi))
    phiComps = C / R
    zComps = ((R - R0) + epsilon * z * np.cos(phi)) / R

    return np.c_[radialComps, phiComps, zComps]


def field_mag(points, field_parameters):
    """
    field_mag(points, field_parameters)
    
    Returns the field magnitude in the form (n, 1) 
    where n is the number of points in R^3
    """

    fieldMag = la.norm(field_comps(points, field_parameters), axis=1)

    return fieldMag


def fieldline_flow(t, y, *field_parameters):
    """
    Returns the vector field for the zeroth-order GCM to obtain the fieldlines
    """

    R0, C, epsilon, initialEnergy = field_parameters

    R = np.sqrt(y[0] ** 2 + y[1] ** 2)
    phi = np.arctan2(y[1], y[0])

    rComp = - (y[2] + epsilon * R * np.cos(phi)) / R
    phiComp = C / R
    zComp = ((R - R0) + epsilon * y[2] * np.cos(phi)) / R

    # convert field components from cylindrical to cartesian 
    xComp = rComp * np.cos(phi) - phiComp * np.sin(phi)
    yComp = rComp * np.sin(phi) + phiComp * np.cos(phi)

    # fieldMag = field_mag([R, phi, y[2]], field_parameters)
    # dJdT = np.sqrt(2.0*(initialEnergy - fieldMag))*fieldMag 

    if np.size(y) == 4:
        fieldMag = field_mag([R, phi, y[2]], field_parameters)
        if np.fabs(initialEnergy - fieldMag) < 1.e-15:  # taking care of a corner case
            fieldMag = initialEnergy
        # print(t,(initialEnergy - fieldMag))
        dJdT = np.sqrt(2.0 * (initialEnergy - fieldMag)) * fieldMag
        return np.append([xComp, yComp, zComp], dJdT)
    else:
        return np.array([xComp, yComp, zComp])


def zgcm_arclength_axisym(t, y, *field_parameters):
    """
    Returns the vector field in terms of the arc-length
    
    Parameters
    ----------
    t : scalar
    arc-length as the independent variable
    
    y : ndarray of size (1, 5)
    [GC position in 3D cartesian, parallel velocity, longitudinal adiabatic invariant] 
    
    Returns
    -------
    out : ndarray of size (5, 1)
    d[GC position in 3D cartesian, parallel velocity, longitudinal adiabatic invariant]/dt
    
    """

    R0, C, epsilon, initialEnergy = field_parameters

    R = np.sqrt(y[0] ** 2 + y[1] ** 2)
    phi = np.arctan2(y[1], y[0])

    rComp = - (y[2] + epsilon * np.cos(phi)) / R
    phiComp = C / R
    zComp = (R - R0) / R

    # convert field components from cylindrical to cartesian 
    xComp = rComp * np.cos(phi) - phiComp * np.sin(phi)
    yComp = rComp * np.sin(phi) + phiComp * np.cos(phi)

    # fieldMag = np.sqrt(C**2 + y[2]**2 + (np.sqrt(y[0]**2 + y[1]**2) - R0)**2)/np.sqrt(y[0]**2 + y[1]**2)
    # print([R, phi, y[2]],rComp)
    fieldMag = la.norm(np.array([xComp, yComp, zComp]))

    if np.abs(t) < 1e-2:  # parallel velocity is small
        xDot = xComp
        yDot = yComp
        zDot = zComp
        vparDot = 0
        # print(initialEnergy - fieldMag)
        jDot = 0  # np.sqrt(2*(initialEnergy - fieldMag))*fieldMag
    else:
        unitVectorField = np.array([xComp, yComp, zComp]) / la.norm(np.array([xComp, yComp, zComp]))

        # dBmag_ds = y[2]/(y[0]**2 + y[1]**2)
        dBmag_ds = dfieldmag_ds(np.array([R, phi, y[2]]), field_parameters)

        xDot = (y[3] * unitVectorField[0]) / fieldMag

        yDot = (y[3] * unitVectorField[1]) / fieldMag

        zDot = (y[3] * unitVectorField[2]) / fieldMag

        vparDot = - dBmag_ds / fieldMag

        # jDot = y[3]**2/(np.sqrt(2)*fieldMag)
        jDot = y[3] ** 2 / fieldMag

    return np.array([xDot, yDot, zDot, vparDot, jDot])


def zgcm_vpar_flow(t, y, *field_parameters):
    """
    Returns the vector field in terms of the parallel velocity in the negative 
    velocity direction
    
    Parameters
    ----------
    t : scalar
    parallel velocity as the independent variable
    
    y : ndarray of size (1, 4)
    [GC position in 3D cartesian, longitudinal adiabatic invariant] 
    
    Returns
    -------
    out : ndarray of size (4, 1)
    d[GC position in 3D cartesian, longitudinal adiabatic invariant]/dt
    
    """

    R0, C, epsilon, initialEnergy = field_parameters

    R = np.sqrt(y[0] ** 2 + y[1] ** 2)
    phi = np.arctan2(y[1], y[0])

    # rComp = - (y[2] + epsilon*np.cos(phi))/R
    # phiComp = C/R
    # zComp = (R - R0)/R 
    rComp = - (y[2] + epsilon * R * np.cos(phi)) / R
    phiComp = C / R
    zComp = ((R - R0) + epsilon * y[2] * np.cos(phi)) / R

    # convert field components from cylindrical to cartesian 
    xComp = rComp * np.cos(phi) - phiComp * np.sin(phi)
    yComp = rComp * np.sin(phi) + phiComp * np.cos(phi)

    unitVectorField = np.array([xComp, yComp, zComp]) / la.norm(np.array([xComp, yComp, zComp]))

    # dBmag_ds = y[2]/(y[0]**2 + y[1]**2)
    dBmag_ds = dfieldmag_ds(np.array([R, phi, y[2]]), field_parameters)

    xDot = -(t * unitVectorField[0]) / dBmag_ds

    yDot = -(t * unitVectorField[1]) / dBmag_ds

    zDot = -(t * unitVectorField[2]) / dBmag_ds

    # jDot = - t**2/(np.sqrt(2)*dBmag_ds)
    jDot = t ** 2 / (dBmag_ds)

    return np.array([xDot, yDot, zDot, jDot])


def zgcm_liftoffsigma(t, y, *field_parameters):
    """
    Vector field to integrate when starting off the sigma^- 
    surface
    """
    R0, C, epsilon, initialEnergy = field_parameters

    R = np.sqrt(y[0] ** 2 + y[1] ** 2)
    phi = np.arctan2(y[1], y[0])

    rComp = - (y[2] + epsilon * np.cos(phi)) / R
    phiComp = C / R
    zComp = (R - R0) / R

    # convert field components from cylindrical to cartesian 
    xComp = rComp * np.cos(phi) - phiComp * np.sin(phi)
    yComp = rComp * np.sin(phi) + phiComp * np.cos(phi)

    fieldMag = la.norm(np.array([xComp, yComp, zComp]))
    # print(fieldMag)

    xDot = xComp
    yDot = yComp
    zDot = zComp
    jDot = np.sqrt(2.0 * (initialEnergy - fieldMag)) * fieldMag

    return np.array([xDot, yDot, zDot, jDot])


def zgcm_vectorfield(t, y, *field_parameters):
    """
    Returns the zeroth-order axisymmetric vector field for the 
    toy tokamak
    
    Parameters
    ----------
    t : scalar
    time-like independent variable
    
    y : ndarray of size (1, 5)
    [GC position in 3D cartesian, parallel velocity, longitudinal adiabatic invariant] 
    
    Returns
    -------
    out : ndarray of size (5, 1)
    d[GC position in 3D, parallel velocity, longitudinal adiabatic invariant]/dt
    """

    R0, C, epsilon, initialEnergy = field_parameters

    R = np.sqrt(y[0] ** 2 + y[1] ** 2)
    phi = np.arctan2(y[1], y[0])

    rComp = - (y[2] + epsilon * np.cos(phi)) / R
    phiComp = C / R
    zComp = (R - R0) / R

    # convert field components from cylindrical to cartesian 
    xComp = rComp * np.cos(phi) - phiComp * np.sin(phi)
    yComp = rComp * np.sin(phi) + phiComp * np.cos(phi)

    fieldMag = la.norm(np.array([xComp, yComp, zComp]))

    unitVectorField = np.array([xComp, yComp, zComp]) / la.norm(np.array([xComp, yComp, zComp]))

    xDot = y[3] * unitVectorField[0]

    yDot = y[3] * unitVectorField[1]

    zDot = y[3] * unitVectorField[2]

    # vParDot = - y[2]/(y[0]**2 + y[1]**2) # Update this expression
    vparDot = - ((epsilon * np.cos(phi) + y[2]) * (
                C ** 2 - C * epsilon * np.sin(phi) + (epsilon * np.cos(phi) + y[2]) ** 2 + (R - R0) ** 2)) / (
                          R ** 2 * (C ** 2 + (epsilon * np.cos(phi) + y[2]) ** 2 + (R - R0) ** 2))

    # jDot = y[3]**2/np.sqrt(2)
    jDot = y[3] ** 2

    return np.array([xDot, yDot, zDot, vparDot, jDot])


def dfieldmag_ds(points, field_parameters):
    """
    Returns the derivative of the field magnitude with respect to 
    arclength
    
    Parameters
    ----------
    points : ndarray of size (n, 3)
    Cylindrical polar coordinates of n points 
    
    field_parameters : ndarray of size (1, 2)
    C and R0 values of the toy tokamak model
    
    """

    R0, C, epsilon = field_parameters[:3]

    if np.size(points, 0) == 3:
        R, phi, z = points
    else:
        R = points[:, 0]
        phi = points[:, 1]
        z = points[:, 2]

    # Unperturbed 
    # modBprime = points[:,2]/points[:,0]**2    

    # Form I perturbation
    modBprime = ((z + epsilon * np.cos(phi)) * (
                C ** 2 + (epsilon * np.cos(phi) + z) ** 2 + (R - R0) ** 2 - C * epsilon * np.sin(phi))) / (
                            R ** 2 * (C ** 2 + (epsilon * np.cos(phi) + z) ** 2 + (R - R0) ** 2))

    return modBprime


def field_mag_analytical(points, field_parameters):
    """
    Returns the field magnitude computed using the analytical 
    expression
    
    Parameters
    ----------
    points : ndarray of size (n, 3)
    Cylindrical polar coordinates of n points 
    
    field_parameters : ndarray of size (1, 2)
    C and R0 values of the toy tokamak model
    
    
    """

    R0, C, epsilon, initialEnergy = field_parameters

    if np.size(points, 0) == 3:
        R, phi, z = points
    else:
        R = points[:, 0]
        phi = points[:, 1]
        z = points[:, 2]

    # fieldMagAnaExp = np.sqrt(C**2 + points[:,2]**2 + \
    #     (points[:,0] - R0)**2)/points[:,0]

    fieldMagAnalyticalExp = np.sqrt((z + epsilon * R * np.cos(phi)) ** 2 + \
                                    C ** 2 + ((R - R0) + epsilon * z * np.cos(phi)) ** 2) / R

    return fieldMagAnalyticalExp


def cart_to_cyl(points, degrees=False):
    """
    Returns the cylindrical polar coordinates for the input
    cartesian coordinates
    """

    rho = np.linalg.norm(points[:, 0:2], axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])

    if degrees:
        theta = np.degrees(theta)

    z = points[:, 2]

    return np.c_[rho, theta, z]


def cyl_to_cart(points):
    """
    Returns the cartesian coordinates for the input cylindrical \
    coordinates
    """

    x = points[:, 0] * np.cos(points[:, 1])
    y = points[:, 0] * np.sin(points[:, 1])
    z = points[:, 2]

    return np.c_[x, y, z]


def plot_torus(resolution, c, a, ax):
    """
    Returns the axis of a torus with major radius, c, and minor radius, a, plot 
    """

    U = np.linspace(0, 1.5 * np.pi, resolution)
    V = np.linspace(0, 2 * np.pi, resolution)
    U, V = np.meshgrid(U, V)
    X = (c + a * np.cos(V)) * np.cos(U)
    Y = (c + a * np.cos(V)) * np.sin(U)
    Z = a * np.sin(V)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax = plt.figure(figsize = (5,5)).add_subplot(projection='3d')
    # fig = plt.figure(figsize=(6,6),dpi=130)
    # ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.set_xlim(-(c + a), (c + a))
    ax.set_ylim(-(c + a), (c + a))
    ax.set_zlim(-2 * a, 2 * a)
    ax.set_box_aspect((1, 1, 1))

    ax.plot_surface(X, Y, Z, alpha=0.7, antialiased=True, \
                    cmap=cm.plasma)
    # ax.plot_surface(x, y, z, antialiased=True, color='orange')
    # plt.show()

    return ax


def dfieldmag_ds_perturb(points, field_parameters):
    """
    CHECK EXPRESSION
    Returns the derivative of the field magnitude with respect to 
    arclength
    
    Parameters
    ----------
    points : ndarray of size (n, 3)
    Cylindrical polar coordinates of n points 
    
    field_parameters : ndarray of size (1, 2)
    C and R0 values of the toy tokamak model
    
    """

    R0, C, epsilon = field_parameters[:3]

    R = points[:, 0]
    phi = points[:, 1]
    z = points[:, 2]

    modBprime = ((z + epsilon * np.cos(phi)) * (
                C ** 2 + (epsilon * np.cos(phi) + z) ** 2 + (R - R0) ** 2 - C * epsilon * np.sin(phi)))

    # modBprime =  ((z + epsilon*np.cos(phi))*(C**2 + (epsilon*np.cos(phi) + z)**2 + (R - R0)**2 - C*epsilon*np.sin(phi)))/(R**2*(C**2 + (epsilon*np.cos(phi) + z)**2 + (R - R0)**2)) # actual function, but causes issues with root solving

    return modBprime


def generate_sigma_minus_axisym(resolution, geom_parameters):
    """
    Returns points on the Sigma minus surface
    defined by 
    
    cylindrical: R < R0 and z = 0 
    cartesian: x^2 + y^2 < R0^2 and z = 0
    """

    R0, r0 = geom_parameters

    theta = np.linspace(0, 2 * np.pi, resolution)  # toroidal angle
    phi = np.linspace(0, 2 * np.pi, resolution)  # poloidal angle
    theta, phi = np.meshgrid(theta, phi)
    xMesh = (majorRad + minorRad * np.cos(phi)) * np.cos(theta)
    yMesh = (majorRad + minorRad * np.cos(phi)) * np.sin(theta)
    zMesh = np.zeros((resolution, resolution))

    idx = np.argwhere(np.sqrt(xMesh ** 2 + yMesh ** 2) < majorRad)
    idx_grid = np.ix_(idx[:, 0], idx[:, 1])

    xMesh = xMesh[idx_grid]
    yMesh = yMesh[idx_grid]
    zMesh = zMesh[idx_grid]

    return xMesh, yMesh, zMesh

# def zgcm_axisym_start_vectorfield(t, y, *field_parameters):
#     """
#     Returns the zeroth-order axisymmetric vector field for the 
#     toy tokamak near the sigma^- surface 
#     """

#     R0, C, epsilon, initialEnergy = field_parameters

#     rComp = - y[2]/np.sqrt(y[0]**2 + y[1]**2)
#     phiComp = C/np.sqrt(y[0]**2 + y[1]**2)
#     zComp = (np.sqrt(y[0]**2 + y[1]**2) - R0)/np.sqrt(y[0]**2 + y[1]**2) 

#     # convert field components from cylindrical to cartesian 
#     phi = np.arctan2(y[1], y[0])
#     xComp = rComp*np.cos(phi) - phiComp*np.sin(phi)
#     yComp = rComp*np.sin(phi) + phiComp*np.cos(phi) 

#     fieldMag = np.sqrt(C**2 + y[2]**2 + \
#                        (np.sqrt(y[0]**2 + y[1]**2) - R0)**2)/np.sqrt(y[0]**2 + y[1]**2)

#     xDot = xComp

#     yDot = yComp

#     zDot = zComp

#     jDot = np.sqrt(initialEnergy - fieldMag)*fieldMag

#     return np.array([xDot, yDot, zDot, jDot])
