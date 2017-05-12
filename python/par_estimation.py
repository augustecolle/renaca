import numpy as np
import scipy.optimize
import pylab as pl

# make fake data for n segments
def getFakeData(n):
    np.random.seed(43237)
    segments = [None]*n # initialize list that will contain n segments
    for i in range(n):
        segment_timesteps  = np.random.randint(5,10) # assume between 5 and 10 timesteps for this segment
        segment_timestamps = np.arange(0,segment_timesteps,1) # s, some "actual" time data
        segment_rho        = 1.   + 0.00*np.random.random(segment_timesteps) # kg/m^3, put some variation on rho
        segment_pm         = 200. + 0.00*(np.random.random(segment_timesteps)-0.5) # W, mechanical power
        segment_vg         = 10. + np.random.normal(0,0.00,segment_timesteps) # m/s, ground speed
        segment_va         =  5. + np.random.normal(0,0.00,segment_timesteps) # m/s, air speed projected on ground speed, can be negative!
        segment_slope      = np.random.normal(0,0.01,segment_timesteps) # dimless, slope, height/distance
        segment = [ segment_timestamps, segment_rho, segment_pm, segment_vg, segment_va, segment_slope ]
        segments[i] = segment
    return segments

# estimate the power given segments
# and the parameters, see errorf for
# explanation about x
def estimatePower(x,segments):
    m    = 100. # kg, mass of cyclist+bicycle
    g    = 9.81 # gravitation
    cyclistUnitPower = 150. # W
    CdA   = x[0]
    Cr    = x[1]
    PUcyc = x[2]
    return np.array([ 0.5*rho*CdA*vg*va**2 + Cr*vg*m*g + m*g*vg*slope - PUcyc*cyclistUnitPower for [_,rho,_,vg,va,slope] in segments ])

# loglikelihood of priors,
# if the variables are independent
# you can multiply the different probabilities
# or equivalently add the log(probabilities)
def lnlikelihoodPriors(CdA,Cr,PUcyc):
    lnprior = 0.

    # CdA prior
    if ( 0.3 <= CdA <= 0.8 ):
        lnprior += np.log(1./(0.8-0.3)) # np.log is ln
    else:
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
        lnprior += np.log( k/l*(PUcyc/l)**(k-1.)*np.exp( -(PUcyc/l)**k) )
    else:
        lnprior += -np.inf
    return lnprior

# error function to minimize
# x is a vector of parameters
# x[0] : CdA, m^2 drag coefficient * A
# x[1] : Cr, dimless rolling coefficient
# x[2] : cyclist power, dimensionless has to be multiplied with "cyclistunitpower" to get SI units
# sigma: W^-1, this is a measure of the allowed discrepancy between known motor power and inferred moter power
#        small sigma: strong fitting, priors are weak, large sigma, loose fitting, priors are strong
def errorf(x,segments,sigma):
    CdA   = x[0]
    Cr    = x[1]
    PUcyc = x[2]
    pm_guess    = estimatePower(x,segments)
    pm_measured = [ s[2] for s in segments ]
    pm_var = sum([ np.sum( (pm_guess[i] - pm_measured[i])**2 ) for i in range(len(segments)) ])
    L = 0.5/sigma**2*pm_var - lnlikelihoodPriors(CdA,Cr,PUcyc)
    #print("Loss function : {:.6e}".format(L))
    return L

def main():
    segments = getFakeData(6)
    x0 = [ 0.6,0.005,1]
    sigma = 0.001 # if this is very small, strong fitting <-> weaker priors. Very large weaker fitting <-> stronger priors
    res = scipy.optimize.minimize(errorf,x0,args=(segments,sigma))
    print("optimal parameters, loss function = {:.6e} ".format(errorf(res.x,segments,sigma)))
    print("succes : {:}".format(res.success))
    print("------------------- ")
    print("| CdA  : {:.3f}     ".format(res.x[0]))
    print("| Cr   : {:.3f}     ".format(res.x[1]))
    print("| Pcyc : {:.2f}     ".format(res.x[2]))
    
    pm_guessf    = np.concatenate(estimatePower(res.x,segments))
    pm_measuredf = np.concatenate([ s[2] for s in segments ])
    fig =pl.figure()
    ax = fig.add_subplot(111)
    ax.plot(pm_guessf,marker='s',color="firebrick",lw=3,ls="dashed",label="fit")
    ax.plot(pm_measuredf,marker='o',color="black",lw=3,ls="solid",label="measured")
    pl.show()

def test():
    segments = getFakeData(6)
    x = [ 0.6,0.005,1]
    Pest = estimatePower(x,segments)
    print(Pest)

if __name__=="__main__":
    main()
    #test()
