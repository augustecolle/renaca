import numpy as np
import scipy.optimize

# make simple container class
# to keep things easily debuggable
# you can always revert to simple
# arrays later if you don't like this...
class Segment:
    def __init__(self,t,rho,pm,vg,va,s):
        self.t = t     # time array
        self.rho = rho # kg/m^3, air density
        self.pm  = pm  # W, mech. power
        self.vg  = vg  # m/s, ground speed
        self.va  = va  # m/s, projected air speed,
        self.s   = s   # dimless, slope, height diff/distance

# make fake data for n segments
def getFakeData(n):
    segments = [None]*n # initialize list that will contain n segments
    for i in range(n):
        segment_timestamps = np.random.randint(5,10) # assume between 5 and 10 timesteps for this segment
        segment_rho        = 1. + 0.2*np.random.random(segment_timesteps) # kg/m^3, put some variation on rho
        segment_pm         = 200. + 50*(np.random.random(segment_timesteps)-0.5) # W, mechanical power
        segment_vg         = 10. + np.random.normal(0,3,segment_timesteps) # m/s, ground speed
        segment_va         = np.random.normal(0,5,segment_timesteps) # m/s, air speed projected on ground speed, can be negative!
        segment_slope      = np.random.normal(0,0.03,segment_timesteps) # dimless, slope, height/distance
        segment = [ segment_timestamps, segment_rho, segment_pm, segment_vg, segment_va, segment_slope ]
        segments[i] = segment
    return segments

# estimate the power given segments
# and the parameters, see errorf for
# explanation about x
def estimatePower(x,segments):
    m    = 100. # kg, mass of cyclist+bicycle
    g    = 9.81 # gravitation
    cyclistUnitPower = 100. # W
    CdA  = x[0]
    Cr   = x[1]
    Pcyc = x[2]
    return np.array([ 0.5*rho*CdA*vg*va**2 + Cr*vg*m*g + m*g*vg*s - Pcyc for [_,rho,_,vg,va,slope] in segments ])

# loglikelihood of priors,
# if the variables are independent
# you can multiply the different probabilities
# or equivalently add the log(probabilities)
def lnlikelihoodPriors(CdA,Cr,PUcyc):
    lnprior = 0.

    # CdA prior
    if ( 0.3 <= CdA <= 0.8 ):
        lnprior += np.log(1./(0.8-0.3)) # np.log is ln
    elsedA  = x[0]
        Cr   = x[1]
            Pcyc = x[2]

        lnprior += -np.inf

    # Cr prior
    if ( 0.00231 <= Cr <= 0.0133 ):
        lnprior += np.log(1./(0.0133-0.00231)) # np.log is ln
    else:
        lnprior += -np.inf

    # P cyclist prior
    k = 1.7
    l = 1.
    if (PUcyc >= 0.):
        lnprior += np.log( k/l*(PUcyc/l)**(k-1.)*np.exp( -(x/l)**k) )
    else:
        lnprior += -np.inf
    return lnprior

# error function to minimize
# x is a vector of parameters
# x[0] : CdA, m^2 drag coefficient * A
# x[1] : Cr, dimless rolling coefficient
# x[2] : cyclist power, dimensionless has to be multiplied with "cyclistunitpower" to get SI units
# x[3] : sigma, W^-1, this is a measure of the allowed discrepancy between known motor power and inferred moter power
#        small sigma: strong fitting, priors are weak, large sigma, loose fitting, priors are strong
def errorf(x,segments):
    CdA  = x[0]
    Cr   = x[1]
    PUcyc = x[2]
    sigma = x[3]
    pm_guess    = estimatePower(x,segments)
    pm_measured = [ s[2] for s in segments ]
    return 0.5/sigma**2*np.sum(pm_measured - pm_guess)**2 - lnlikelihoodPriors(CdA,Cr,Pcyc)

def main():
    segments = getFakeData(10)
    x0 = [ 0.6,0.005,1]
    res = scipy.optimize.minimize(errorf,x0
