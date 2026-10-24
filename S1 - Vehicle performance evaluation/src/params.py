import numpy as np

# --- ENVIRONMENT ---
RHO_AIR        = 1.225      # [kg/m^3] air density, ISA sea level
G              = 9.81       # [m/s^2] gravitational acceleration

# --- VEHICLE ---
MASS           = 1685       # [kg] vehicle mass - coursework brief
MASS_FACTOR    = 1.3        # [-] rotating inertia factor, m_eff = MASS_FACTOR*MASS (assumed)
V_LIMIT        = 250/3.6    # [m/s] electronic speed limiter (assumed - replace with your value)

# --- AERO ---
CD             = 0.36       # [-] drag coefficient - coursework brief
AREA_FRONT     = 2.3        # [m^2] frontal area - coursework brief
CL             = 0.2        # [-] lift coefficient (positive = lift) - coursework brief

# --- TYRES ---
WHEEL_RADIUS   = 0.343      # [m] rolling radius - coursework brief
C_RR           = 0.0085     # [-] rolling resistance coefficient - coursework brief
MU             = 0.9        # [-] tyre-road friction coefficient (assumed, dry asphalt)

# --- POWERTRAIN ---
RPM            = np.array([2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000])   # [rpm] - engine datasheet
TORQUE         = np.array([280, 331, 379, 429, 459, 466, 470, 476, 480, 480, 475, 471, 470, 467, 462, 455, 445, 433, 421, 402, 368])                        # [Nm] - engine datasheet
POWER          = np.array([ 59,  78,  99, 124, 144, 159, 172, 187, 201, 214, 224, 234, 246, 257, 266, 274, 279, 284, 286, 284, 269])                        # [kW] - engine datasheet, for checking only
REV_LIMIT      = 7000       # [rpm] end of the torque map
GEAR_RATIOS    = np.array([3.563, 2.526, 1.679, 1.022, 0.788, 0.761, 0.635])   # [-] - gearbox datasheet
FINAL_DRIVE    = np.array([4.059, 3.450, 3.450, 4.059, 4.059, 3.450, 3.450])   # [-] per gear (dual-clutch, two output shafts)
DRIVELINE_EFF  = 0.85       # [-] driveline efficiency
SHIFT_TIME     = 0.008      # [s] torque gap during a shift

# --- CROSSWIND ---
WIND           = 0.0        # [m/s] crosswind speed, perpendicular to travel (0 = still air)
PSI            = np.array([0,4,8,12,15])                   # [deg] yaw angle
CL_PSI         = np.array([0.20, 0.21, 0.26, 0.34, 0.41])  # [-] lift coefficient at each yaw angle
CD_PSI         = np.array([0.36, 0.37, 0.41, 0.44, 0.46])  # [-] drag coefficient at each yaw angle