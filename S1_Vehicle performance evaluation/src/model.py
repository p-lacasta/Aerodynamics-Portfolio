import numpy as np
import params as p

# All functions take v in [m/s] and work on a single value or a numpy array.
# Gears are numbered 1 to 7.


# --- ENGINE ---

def engine_torque(rpm):
    """Engine torque [Nm] from the map. Outside 2000-7000 rpm np.interp holds the end values
    (below 2000 rpm this stands in for clutch slip at launch)."""
    return np.interp(rpm, p.RPM, p.TORQUE)

def engine_rpm(v, gear):
    """Engine speed [rpm] at road speed v in a gear."""
    ratio = p.GEAR_RATIOS[gear-1] * p.FINAL_DRIVE[gear-1]
    return v/p.WHEEL_RADIUS * ratio * 60/(2*np.pi)


# --- AERO (with crosswind p.WIND) ---

def yaw_angle(v):
    """Yaw angle [deg] of the relative wind. 0 in still air."""
    return np.degrees(np.arctan2(p.WIND, v))

def air_speed_sq(v):
    """Square of the airspeed seen by the car [m^2/s^2]."""
    return v**2 + p.WIND**2

def drag_coeff(v):
    """C_d at the current yaw angle: baseline CD plus the rise from the CD_PSI table.
    Beyond the table (low speed in a crosswind) np.interp holds the last value."""
    return p.CD + np.interp(yaw_angle(v), p.PSI, p.CD_PSI) - p.CD_PSI[0]

def lift_coeff(v):
    """C_L at the current yaw angle, built the same way from the CL_PSI table."""
    return p.CL + np.interp(yaw_angle(v), p.PSI, p.CL_PSI) - p.CL_PSI[0]


# --- RESISTANCE FORCES ---

def drag_force(v):
    """Drag along the direction of travel [N] (C_d taken in body axes)."""
    return 0.5*p.RHO_AIR*drag_coeff(v)*p.AREA_FRONT*air_speed_sq(v)

def lift_force(v):
    return 0.5*p.RHO_AIR*lift_coeff(v)*p.AREA_FRONT*air_speed_sq(v)

def normal_force(v):
    """Total tyre load [N]: weight minus aero lift (level road)."""
    return p.MASS*p.G - lift_force(v)

def rolling_force(v):
    return p.C_RR*normal_force(v)

def resistance_force(v):
    return drag_force(v) + rolling_force(v)


# --- TRACTIVE FORCE ---

def tractive_force(v, gear):
    """Force at the wheels [N] at full throttle in a gear, capped by tyre grip."""
    ratio = p.GEAR_RATIOS[gear-1] * p.FINAL_DRIVE[gear-1]
    F_engine = engine_torque(engine_rpm(v, gear)) * ratio * p.DRIVELINE_EFF / p.WHEEL_RADIUS
    F_grip = p.MU * normal_force(v)
    return np.minimum(F_engine, F_grip)


# --- EQUATION OF MOTION ---

def acceleration(v, gear, drive=True):
    """dv/dt [m/s^2]. drive=False gives the coast-down during a shift (torque gap)."""
    F_t = tractive_force(v, gear) if drive else 0.0
    return (F_t - resistance_force(v)) / (p.MASS_FACTOR*p.MASS)
