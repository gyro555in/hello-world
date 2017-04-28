
"""Packages"""
#from __future__ import division, print_function
import numpy.random as rnd
import numpy, scipy, pylab, math, sympy, mpmath, time, sys, matplotlib.pyplot as plt 
from mpmath import mp 
from scipy import integrate
from scipy import fftpack
import vegas
"""mpmath for higher precision tolerances"""
start_time = time.time() 
"""Set timer"""

"""Constants in atomic units"""
h=1.0; """Planck constant"""
m=0.5; """electron mass"""
e=math.sqrt(2); """electron charge"""
eps=0.25/(math.pi); """permitivity of vacuum"""
a_b=1.0; """Bohr radius """
K_b=b=1.0; """Boltzmann constant"""
#w_in = numpy.linspace(0.04,0.24,8);
#p_in = 0.05 ;
"""Input Parameters"""
electron_temperature = input('electron temperature in eV :   ')
electron_density = input('electron density in m^-3 :   ')
k_vector = input('k vector in m^-1 :   ')

"""Plasma Parameters"""
T = Tm = electron_temperature*1.0/13.6 ; """electron temperature"""
n_e = (electron_density)*(10**-30)*(0.5291**3); """electron density"""
q = k = k_vector*(0.5291)/(10**10); """k-vector"""
T_f = t = 1.0*(h*h/(2*m*K_b))*( ( 3*math.pi*math.pi*n_e )**((2.0/3))); """Fermi temperature"""
k_f = (2.0*m*K_b*T_f/(h*h))**(0.5);"""Fermi K-vector"""
w_pl = (n_e*4*math.pi*e*e/(m))**(0.5);"""Plasma frequency"""
Lambda_D = ( n_e / Tm )**0.5 ; """Debye inverse screening length"""
w_q =  ( (w_pl * ( 1.0 + (q/k)**2 ) ) + (q**4) )**0.5 ; """Plasmon satellite frequency"""

"""Outputs plasma parameters"""
def plasma_parameters():
  print 'Plasma parameters in atomic units'
  print 'Electron temperature :   ' ,  Tm 
  print 'Plasma frequency     :   ' ,  w_pl
  print 'Electron density     :   ' ,  n_e
  print 'K-vector             :   ' ,  k
  print 'Fermi temperature    :   ' ,  T_f 
  print 'Debye inverse screening length :   ' ,  Lambda_D
  print 'Chemical potential   :   ' ,  mu
#######################################################################################################
"""Chemical potential"""
cpcheck2 = 2.0*math.pi*math.pi*n_e;
errorcheck=[];
chempot=numpy.linspace(-50*Tm,50*Tm,50000);
#Loop to check the right chemical potential
def cp(p):
  return 2.0*(p**2)/(1.0 + scipy.exp( ( (p**2) - chempot[i])/(K_b*Tm) ) ); 
"""Find the convergence by minimizing the difference between numerical and analytical solution"""
for i in range(len(chempot)):
  numans = scipy.integrate.quad(cp,0,numpy.inf)[0]
  errorcheck.append( abs(numans - cpcheck2) )
  #if abs(numans - cpcheck2) >= 10**(-03)  and abs(numans - cpcheck2) <= 10**(-02)  :
  #print "error = %f  chemical potential = %f " %(abs(numans - cpcheck2), chempot[i])
#gives the index of least error:  errorcheck.index(min(errorcheck))
g = gamma = mu = chempot[ errorcheck.index(min(errorcheck))  ];
###############################################################################################################################
w=numpy.linspace(0,5,50); """Freqeuncy range"""
gg = numpy.linspace(0.0,5.0,1000);
imvalues=[];
re_rpa=[];

def intimrpa(w):
	return (2.0/math.pi)*w*(1.0/( (w**2) - (gg[i]**2) ) )*(2.0*m*e*e*h*h/(q**3))*(1.0*T*m*scipy.log(1.0*scipy.exp((-g + 0.5*(-1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 1.0*T*m*scipy.log(1.0*scipy.exp((-g + 0.5*(1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 0.5*(-1.0*h*m*w/q + 0.5*q)**2 + 0.5*(1.0*h*m*w/q + 0.5*q)**2)

"""Imaginary part of dielectric function as a function of frequency"""
def imrpa(w):
	return (2.0*m*e*e*h*h/(q**3))*(1.0*T*m*scipy.log(1.0*scipy.exp((-g + 0.5*(-1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 1.0*T*m*scipy.log(1.0*scipy.exp((-g + 0.5*(1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 0.5*(-1.0*h*m*w/q + 0.5*q)**2 + 0.5*(1.0*h*m*w/q + 0.5*q)**2)

"""Real part of dielectric function as a function of momentum"""
def re_rpa_fn(p):
	return (4.0*0.5*m*e*e/(math.pi*h*q*q*q))*( scipy.log( (p*a_b/h) - (0.5*q*a_b/h) - (m*w[i]*a_b/q) ) + scipy.log( (p*a_b/h) - (0.5*q*a_b/h) + (m*w[i]*a_b/q) ) )*p*(1.0/(1.0 + scipy.exp( ((0.5*p*p/m) - gamma)/(K_b*T))))
	
def re_rpa_fn2(p):
	return (4.0*0.5*m*e*e/(math.pi*h*q*q*q))*( scipy.log( (p*a_b/h) - (0.5*q*a_b/h) - (m*w*a_b/q) ) + scipy.log( (p*a_b/h) - (0.5*q*a_b/h) + (m*w*a_b/q) ) )*p*(1.0/(1.0 + scipy.exp( ((0.5*p*p/m) - gamma)/(K_b*T))))	

"""Imaginary part of dielectric function at given frequencies"""
#for i in range(len(w)):i
#	imvalues.append(imrpa(w[i]))
imvalues = map( imrpa, w );
	
"""Real part of dielectric function at given frequencies"""
for i in range(len(w)):
  re_rpa.append( 1.0 - scipy.integrate.quad(re_rpa_fn, -numpy.inf,  numpy.inf)[0] )
#re_rpa = map( 1.0 - scipy.integrate.quad(re_rpa_fn2, - 10**4, 10**4 )[0] , w  );  
  
def  dielectric_function():
 plt.plot(w, imvalues,  label=" $ \Im  $" )
 plt.plot(w, re_rpa,  label=" $ \Re  $" )
 plt.xlabel('E (Ha)')
 plt.ylabel('$  \epsilon  $ ') 
 plt.legend()
 pylab.show()
###############################################################################################################################
"""Define a signum function"""
def signum(x):
 if x > 0 :
  return 1.0; 
 if x < 0 :
  return -1.0;
 if x==0:
  return 0.0;  
###############################################################################################################################
"""Kramers-Kronig relationship using Hilbert transform"""
hs1 = scipy.fftpack.hilbert(imvalues); """Real part of dielectric function using Hilbert transform """
hs2 = -scipy.fftpack.hilbert(re_rpa);   """Imaginary part of dielectric function using Hilbert transform """
def kramerskronig_hilbert():
 plt.plot(w, hs1,  label=" $ \Re  $" )
 plt.plot(w, hs2,  label=" $ \Im  $" )
 plt.xlabel('E (Ha)')
 plt.ylabel('$  \epsilon  $ ') 
 plt.legend()
 pylab.show()
###############################################################################################################################
"""Inverse of dielectric function at given frequencies"""
def inverse_dielectric_function():
 inv_im = [];
 inv_re = [];
 for i in range(len(w)):
  inv_re.append( re_rpa[i] / ( (re_rpa[i])**2 + (imvalues[i])**2 ) )
 for i in range(len(w)):
  inv_im.append( - imvalues[i] / ( (re_rpa[i])**2 + (imvalues[i])**2 ) ) 

 plt.plot(w, inv_im,  label=" $ \Im  $" )
 plt.plot(w, inv_re,  label=" $ \Re  $" )
 plt.xlabel('E (Ha)')
 plt.ylabel('$  \epsilon^{-1}  $ ') 
 plt.legend()
 pylab.show() 
###############################################################################################################################
"""mpmath conversion for using Hilbert transforms for any domain"""
"""Imaginary part of dielectric function as a function of frequency using mpmath precision"""
#def imrpa_mpmath(w):
#	return (2.0*m*e*e*h*h/(q**3))*(1.0*T*mp.log(1.0*mp.exp((-g + 0.5*(-1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 1.0*T*m*mp.log(1.0*mp.exp((-g + 0.5*(1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 0.5*(-1.0*h*m*w/q + 0.5*q)**2 + 0.5*(1.0*h*m*w/q + 0.5*q)**2);
	
"""Real part of dielectric function as a function of frequency using mpmath precision"""
#re_rpa_fn_mpmath = lambda p: (4.0*0.5*m*e*e/(mp.pi*h*q*q*q))*( mp.log( (p*a_b/h) - (0.5*q*a_b/h) - (m*w*a_b/q) ) + #mp.log( (p*a_b/h) - (0.5*q*a_b/h) + (m*w*a_b/q) ) )*p*(1.0/(1.0 + mp.exp( ((0.5*p*p/m) - gamma)/(K_b*T))))

"""dielectric function at given frequencies using mpmath precision"""
def dielectric_function_mpmath(decimalprecision):
 """set precision values of decimal places"""
 re_rpa_fn_mpmath = lambda p: (4.0*0.5*m*e*e/(mp.pi*h*q*q*q))*( mp.log( (p*a_b/h) - (0.5*q*a_b/h) - (m*w[i]*a_b/q) ) + mp.log( (p*a_b/h) - (0.5*q*a_b/h) + (m*w[i]*a_b/q) ) )*p*(1.0/(1.0 + mp.exp( ((0.5*p*p/m) - gamma)/(K_b*T)))) 
 imrpa_mpmath = lambda w: (2.0*m*e*e*h*h/(q**3))*(1.0*T*mp.log(1.0*mp.exp((-g + 0.5*(-1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 1.0*T*m*mp.log(1.0*mp.exp((-g + 0.5*(1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 0.5*(-1.0*h*m*w/q + 0.5*q)**2 + 0.5*(1.0*h*m*w/q + 0.5*q)**2)
 mp.dps = decimalprecision; 
 w = numpy.linspace(-5,5,50);
 imvalues_mpmath=[];
 re_rpa_mpmath=[];
 rempval=[];
 for i in range(len(w)):
	imvalues_mpmath.append(imrpa_mpmath(w[i]));
 for i in range(len(w)):
  re_rpa_mpmath.append( 1.0 - mp.quad(re_rpa_fn_mpmath, [-mp.inf,  mp.inf] ))
 """Conversion to float format from mpc format using $ float(mp.nstr(mpmath.mpc(re_rpa_mpmath[0]).real)) $ """ 
 for i in range(len(w)): 
  rempval.append( float(  mp.nstr(  mpmath.mpc( re_rpa_mpmath[i]  ).real ) ) ) 
  
 hs1_im_mp =  -scipy.fftpack.hilbert(rempval) ;
 hs1_re_mp =  scipy.fftpack.hilbert(imvalues_mpmath) ;
 plt.plot(w, imvalues_mpmath,  label=" $ \Im  $" )
 plt.plot(w, rempval,  label=" $ \Re  $" )
 plt.xlabel('E (Ha)')
 plt.ylabel('$  \epsilon  $ ') 
 plt.legend()
 pylab.show() 
########################################################################################################################
"""Kramers-Kronig relationship using Hilbert transform with mpmath precision"""
def dielectric_function_mpmath_hilbert(decimalprecision) :
 re_rpa_fn_mpmath = lambda p: (4.0*0.5*m*e*e/(mp.pi*h*q*q*q))*( mp.log( (p*a_b/h) - (0.5*q*a_b/h) - (m*w[i]*a_b/q) ) + mp.log( (p*a_b/h) - (0.5*q*a_b/h) + (m*w[i]*a_b/q) ) )*p*(1.0/(1.0 + mp.exp( ((0.5*p*p/m) - gamma)/(K_b*T))))
 mp.dps = decimalprecision; 
 imrpa_mpmath = lambda w: (2.0*m*e*e*h*h/(q**3))*(1.0*T*mp.log(1.0*mp.exp((-g + 0.5*(-1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 1.0*T*m*mp.log(1.0*mp.exp((-g + 0.5*(1.0*h*m*w/q + 0.5*q)**2/m)/T) + 1.0) - 0.5*(-1.0*h*m*w/q + 0.5*q)**2 + 0.5*(1.0*h*m*w/q + 0.5*q)**2)
 w = numpy.linspace(-1,1,500);
 imvalues_mpmath=[];
 re_rpa_mpmath=[];
 rempval=[];
 for i in range(len(w)):
	imvalues_mpmath.append(imrpa_mpmath(w[i]));
 for i in range(len(w)):
  re_rpa_mpmath.append( 1.0 - mp.quad(re_rpa_fn_mpmath, [-mp.inf,  mp.inf ] ))
 """Conversion to float format from mpc format using $ float(mp.nstr(mpmath.mpc(re_rpa_mpmath[0]).real)) $ """ 
 for i in range(len(w)): 
  rempval.append( float(  mp.nstr(  mpmath.mpc( re_rpa_mpmath[i]  ).real ) ) ) 
 hs1_im_mp =  -scipy.fftpack.hilbert(rempval) ;
 hs1_re_mp =  scipy.fftpack.hilbert(imvalues_mpmath) ;
 plt.plot(w, hs1_im_mp,  label=" $ \Im - HT  $" )
 plt.plot(w, hs1_re_mp,  label=" $ \Re - HT  $" )
 plt.xlabel('E (Ha)')
 plt.ylabel('$  \epsilon  $ ') 
 plt.legend()
 pylab.show() 

########################################################################################################################

############################################################################################################
"""imaginary part of dielectric function as a function of frequency and momentum"""
imrpa2 = lambda q_grid, w : (2.0*m*e*e*h*h/(q_grid**3))*(1.0*T*m*scipy.log(1.0*scipy.exp((-g + 0.5*(-1.0*h*m*w/q_grid + 0.5*q_grid  )**2/m )/T) + 1.0) - 1.0*T*m*scipy.log(1.0*scipy.exp((-g + 0.5*(1.0*h*m*w/q_grid + 0.5*q_grid)**2/m)/T)  + 1.0) - 0.5*(-1.0*h*m*w/q_grid + 0.5*q_grid)**2 + 0.5*(1.0*h*m*w/q_grid + 0.5*q_grid)**2)

"""Real part of dielectric function as a function of momentum"""
re_rpa_grid = lambda p : (4.0*0.5*m*e*e/(math.pi*h*q_grid[j]*q_grid[j]*q_grid[j]))*( scipy.log( (p*a_b/h) - (0.5*q_grid[j]*a_b/h) - (m*w[i]*a_b/q_grid[j]) ) + scipy.log( (p*a_b/h) - (0.5*q_grid[j]*a_b/h) + (m*w[i]*a_b/q_grid[j]) ) )*p*(1.0/(1.0 + scipy.exp( ((0.5*p*p/m) - gamma)/(K_b*T))))
	
"""Generate momentum grid"""
q_grid=numpy.linspace(0.25 * (10**9)* (0.5291)/(10**10) , (2.00 * 10**9)*(0.5291)/(10**10) ,2); """Momentum grid"""
 
"""Store imaginary values as an array """
imvalues_grid=[];
imvalues_grid2=[];
for i in range( len(q_grid)  ):
 for j in range( len(w) ):
  imvalues_grid.append( imrpa2( q_grid[i], w[j]  ) ) ;
  
imvalues_grid2 = numpy.split( numpy.asarray( imvalues_grid) , len(q_grid)  )  
""" splits list into chucks of arrays for each momenta """ 

"""Store real values as an array """ 
revalues_grid=[];
revalues_grid2=[];
for j in range( len(q_grid)  ):
 for i in range( len(w) ):
  revalues_grid.append( 1.0 - scipy.integrate.quad(re_rpa_grid, -numpy.inf,  numpy.inf)[0]     ) ;
  
revalues_grid2 = numpy.split( numpy.asarray( revalues_grid) , len(q_grid)  ) 
""" splits list into chucks of arrays for each momenta """ 

"""imaginary part of dielectric function in a grid"""
def dielectric_function_grid_im():
 for i in range(len(q_grid)):
  plt.plot(w,  imvalues_grid2[i]   )
  plt.xlabel('E (Ha)')
  plt.ylabel('$ \Im \epsilon  $ ') 
  plt.legend()
 pylab.show()
 
"""real part of dielectric function in a grid"""
def dielectric_function_grid_re():
 for i in range(len(q_grid)):
  plt.plot(w,  revalues_grid2[i]   )
  plt.xlabel('E (Ha)')
  plt.ylabel('$ \Re \epsilon  $ ') 
  plt.legend()
 pylab.show()
############################################################################################# 
""" Hartree-Fock term frequency independent"""
hartreefock  = lambda p : scipy.integrate.dblquad(  lambda q_vector, theta : - 1.0/( 1.0 + scipy.exp( ( p**2 + q_vector**2  - 2*p*q_vector*scipy.cos(theta) - gamma  )/Tm  )    )
 , -0.5*scipy.pi, 0.5*scipy.pi, lambda theta: 0, lambda theta: numpy.inf )[0] ;
#############################################################################################
""" chemical potential determination from spectral function """
n_e_iter = lambda p , w : ( 4.0*scipy.pi*(p**2))*( 0.125/ (scipy.pi)**3 )*((2.0 * scipy.pi)**0.5)/(c)*w*scipy.exp(( -( w - (p**2) - hartreefock(p)  )**2 )/ (2.0 * (c**2) ))*( 1.0 /  ( 1.0 + scipy.exp( (w - g_check)/Tm ) )   )
c=1.0;
#scipy.integrate.dblquad( n_e_iter, -2 , 2, lambda w: 0, lambda w : 1 )[0]
#Loop to check the right chemical potential
"""Find the convergence by minimizing the difference between numerical and analytical solution"""
def chemical_potential():
 n_e_iter = lambda p , w : ( 4.0*scipy.pi*(p**2))*( 0.125/ (scipy.pi)**3 )*((2.0 * scipy.pi)**0.5)/(c)*w*scipy.exp(( -( w - (p**2) - hartreefock(p)  )**2 )/ (2.0 * (c**2) ))*( 1.0 /  ( 1.0 + scipy.exp( (w - chempot_iter[i])/Tm ) )   )
 c = 1.0; 
 errorcheck_n_e=[];
 chempot_iter=numpy.linspace(0.0,2.0,10);
 for i in range(len(chempot_iter)):
  numans_iter = scipy.integrate.dblquad( n_e_iter, -numpy.inf ,numpy.inf, lambda w: 0, lambda w : 1 )[0] 
 errorcheck_n_e.append( abs(numans_iter - n_e) );
 g_iter = gamma_iter = mu_iter = chempot_iter[ errorcheck_n_e.index(min(errorcheck_n_e))  ];
 print g_iter
#############################################################################################
""" Spectral function: First moment"""
""" c_p is normlaized to unity """
c_p = lambda p : 3*(p**0)   
sp_fn_moment1 = lambda p :  scipy.integrate.quad( lambda w: (w)*(0.5/scipy.pi)*(((2.0 * scipy.pi)**0.5)/( c_p(p) ) )*scipy.exp(( -( w - p**2 - hartreefock(p)  )**2 )/ (2.0 *c_p(p)**2)) , -numpy.inf, numpy.inf  )[0]
sp_fn_moment2 = lambda p :  scipy.integrate.quad( lambda w: (w**2)*(0.5/scipy.pi)*(((2.0 * scipy.pi)**0.5)/( c_p(p) ) )*scipy.exp(( -( w - p**2 - hartreefock(p)  )**2 )/ (2.0 *c_p(p)**2)) , -numpy.inf, numpy.inf  )[0]
sp_fn_moment3 = lambda p :  scipy.integrate.quad( lambda w: (w**3)*(0.5/scipy.pi)*(((2.0 * scipy.pi)**0.5)/( c_p(p) ) )*scipy.exp(( -( w - p**2 - hartreefock(p)  )**2 )/ (2.0 *c_p(p)**2)) , -numpy.inf, numpy.inf  )[0]
"""E_1st_moment(p) is the first moment RHS as a function of momentum """
E_1st_moment = lambda p: (p**2) + hartreefock(p)
E_2nd_moment = lambda p: ( (p**2) + hartreefock(p) )**2
E_3rd_moment = lambda p: ( (p**3) + hartreefock(p) )**3
p_range = numpy.linspace(0,1,5);
#############################################################################################
""" Spectral function w.r.t. frequency plot at fixed momentum """
""" Works at small momentum p """
spec_fn_p_w = lambda  p, w : ( ( (2.0 * scipy.pi)**0.5 ) /( c_p(p) ) ) * scipy.exp( ( -( w - (p**2) - hartreefock(p)  )**2 )/ (2.0 * (c_p(p)**2) ) ) 
spec_fn_norm = lambda p :  scipy.integrate.quad( lambda w : (0.5/scipy.pi)*( ( (2.0 * scipy.pi)**0.5 ) /( c_p(p) ) ) * scipy.exp( ( -( w - (p**2) - hartreefock(p)  )**2 )/ (2.0 * (c_p(p)**2) ) ) , -numpy.inf, numpy.inf )[0]
def spec_fn_freq_plot(p):
  plt.plot( w,  map( spec_fn_p_w, w , p * numpy.ones( len(w) )    )  )
  pylab.show()

#############################################################################################
"""Plots frequency independent hartree-fock self-energy"""
def hartree_fock_self_energy(p1,p2,np) :
 q_v = numpy.linspace(p1,p2,np);
 hfval = map( hartreefock, q_v  );
 plt.plot(q_v, hfval)
 pylab.show()
#############################################################################################
"""mpmath integrator for spectral function"""
mp.dps = 25;
spec_fn_norm_mpmath = lambda p :  mp.quad( lambda w : (0.5/mp.pi)*( ( (2.0 * mp.pi)**0.5 ) /( c_p(p) ) ) * mp.exp( ( -( w - (p**2) - hartreefock(p)  )**2 )/ (2.0 * (c_p(p)**2) ) ) , [-mp.inf, mp.inf] )
#############################################################################################
""" GW """
"""Imaginary part of rpa dielectric function as a function of frequency and momentum"""
def im_rpa_pw( p, w ) :
 return (2.0*m*e*e*h*h/(p**3))*(1.0*T*m*scipy.log(1.0*scipy.exp((-g + 0.5*(-1.0*h*m*w/p + 0.5*p)**2/m)/T) + 1.0) - 1.0*T*m*scipy.log(1.0*scipy.exp((-g + 0.5*(1.0*h*m*w/p + 0.5*p)**2/m)/T) + 1.0) - 0.5*(-1.0*h*m*w/p + 0.5*p)**2 + 0.5*(1.0*h*m*w/p + 0.5*p)**2)
"""Real part of rpa dielectric function as a function of frequency and momentum in the right form which is passed after integration """
re_rpa_pw = lambda p, w : 1.0 - scipy.integrate.quad( lambda y : (4.0*0.5*m*e*e/(math.pi*h*p*p*p))*( scipy.log( (y*a_b/h) - (0.5*p*a_b/h) - (m*w*a_b/p) ) + scipy.log( (y*a_b/h) - (0.5*p*a_b/h) + (m*w*a_b/p) ) )*y*(1.0/(1.0 + scipy.exp( ((0.5*y*y/m) - gamma)/(K_b*T))))	, -3, 3 )[0]
""" ########################### """
n_bose_back = lambda w: 1.0 / ( scipy.exp( (w - gamma)/Tm ) -  1.0) 
n_bose_forw = lambda w: 1.0 + ( 1.0 / ( scipy.exp( (w - gamma)/Tm ) -  1.0) )
n_fermi_forw = lambda p:  1.0 / ( scipy.exp( ( (p**2) - gamma)/Tm )   + 1.0) 
n_fermi_back = lambda p:  1.0 - (1.0 / ( scipy.exp( ( (p**2) - gamma)/Tm )   + 1.0) ) 
greensfn_forw = lambda p, w : spec_fn_p_w(p,w) * n_fermi_forw(p)
greensfn_back =  lambda p, w : spec_fn_p_w(p,w) * n_fermi_back(p)
V_non_int = lambda p : 4.0 *scipy.pi /(p**2) 
W_forw = lambda p, w :  V_non_int(p)* (1.0 + n_bose_forw(w)) * (im_rpa_pw(p,w)/( (im_rpa_pw(p,w))**2 + (re_rpa_pw(p,w))**2  ))
W_back =  lambda p, w : V_non_int(p)* n_bose_back(w) * (im_rpa_pw(p,w)/( (im_rpa_pw(p,w))**2 + (re_rpa_pw(p,w))**2  ))
#self_energy_im_forw = lambda p, w : greensfn_forw(p,w) * W_forw(p,w)
#self_energy_im_back =  lambda p, w :  greensfn_back(p,w) * W_back(p,w)
#im_self_energy = lambda p, w: scipy.integrate.dblquad(  - (1.0 / (2.0 * scipy.pi )**4 )*( self_energy_im_forw(p,w) - self_energy_im_back(p,w) ) )
self_energy_im_forw = lambda p, w : scipy.integrate.dblquad( lambda p_dummy,w_dummy : greensfn_forw( p_dummy , w_dummy  ) * W_forw( p - p_dummy, w - w_dummy),0,0.1, lambda w_dummy:0,lambda w_dummy:0.1  )[1]
self_energy_im_back = lambda p, w : scipy.integrate.dblquad( lambda p_dummy,w_dummy : greensfn_back( p_dummy , w_dummy  ) * W_back( p - p_dummy, w - w_dummy),0,2, lambda w_dummy:-1,lambda w_dummy:1  )[1]
im_self_energy = lambda p, w: self_energy_im_forw(p,w) - self_energy_im_back(p,w)
############################################################################################################
""" Montroll-Ward Self-energy """
def self_energy_montrollward( p_in, w_in ) :
 return scipy.integrate.tplquad( lambda  p, w, theta :
(-2.0/(scipy.pi)**2)*(im_rpa_pw(p,w)/( (im_rpa_pw(p,w))**2 + (re_rpa_pw(p,w))**2  ))*(1.0/( w_in - w - p_in**2 - p**2 + 2*p*p_in*scipy.cos(theta) ) ) * ( n_bose_forw(w) - (1.0/(scipy.exp(p_in**2 + p**2 - 2*p*p_in*scipy.cos(theta) - gamma)/Tm )  + 1.0 ) ) , 1.0 , 0.5*scipy.pi,  lambda theta : -0.5 , lambda theta : 0.5 , lambda theta , w: 0.1, lambda theta, w: 0.15  )[1]   
#Needs two arguments for calling and one dummy for integrating over angles
############################################################################################################
#def self_energy_montrollward_plot( w_min, w_max, number_w_points, momentum_value):
#  w_range = numpy.linspace( w_min, w_max, number_w_points  );
#  plt.plot( w_range ,  map( self_energy_montrollward, w_range , momentum_value * numpy.ones( len(w_range) )    )  )
#  pylab.show()
############################################################################################################
""" Available Function calls"""  
""" plasma_parameters() """  """Displays plasma parameters"""
""" dielectric_function() """   """Plots dielectric function vs frequency"""
""" inverse_dielectric_function() """   """Plots inverse of dielectric function vs frequency"""
""" dielectric_function_mpmath(int(decimalplaces)) """   """Plots dielectric function vs frequency using mpmath precision"""
""" kramerskronig_hilbert() """ """Plots dielectric function vs frequency using Hilbert transform """
""" dielectric_function_mpmath_hilbert(int(decimalplaces)) """   """Plots dielectric function vs frequency using Hilbert transform with  mpmath precision"""
"""dielectric_function_grid_im()""" """ Plots imaginary part of dielectric function vs frequency for different momenta """
"""dielectric_function_grid_re()""" """ Plots real part of dielectric function vs frequency for different momenta """
""" """
############################################################################################################
print 'Available function calls'
print 'plasma_parameters() '
print 'dielectric_function() '
print 'inverse_dielectric_function() '
print 'dielectric_function_mpmath(number of decimal places for precision)'
print 'kramerskronig_hilbert()'
print 'dielectric_function_mpmath(number of decimal places for precision)'
print 'dielectric_function_grid_im() '
print 'dielectric_function_grid_re()  '
print 'hartree_fock_self_energy(min. momentum, max. momentum,no. of points) '
print 'spec_fn_freq_plot( fixed momentum  ) '
print 'self_energy_montrollward_plot(   ) '
#### https://www.hzdr.de/CGI/checklm?mathematica
######################################################################################################################
"""Runtime"""
print 'runtime : ',  time.time() - start_time , 's'
#############################################################################################
# Numerically integrate f using trapezoid rule for 1d domain
# f is the function, a,b - domain, N-no. of intervals  
def Trapezoid(f,a,b,N):

	if a > b:
		print 'Error: a=%g > b=%g' % (a,b)
		return None

	h = (b-a)/float(N)
	
	integral=0
	
	# Syntax.
	# for k in range(start,stop,step):
	# NOTE: Need colon and indentation just as for functions.
	# Default step is 1, so can do
	# for k in range(start,stop)
	
	# YOU MUST KNOW THIS: 
        # for k in range(1,4) loops over k=1,2,3. 
	# k=4 is NOT included!!!
      
        N=int(round(N)) #Make sure N is an integer for loop
        	
	for k in range(0,N): 
		integral += f(a+k*h) + f(a+(k+1)*h)
	
	integral *= h/2.0
	
	return integral
################################################
#2d trapezoidal
def trapz2d(z,x=None,y=None,dx=1.,dy=1.):
    ''' Integrates a regularly spaced 2D grid using the composite trapezium rule. 
    IN:
       z : 2D array
       x : (optional) grid values for x (1D array)
       y : (optional) grid values for y (1D array)
       dx: if x is not supplied, set it to the x grid interval
       dy: if y is not supplied, set it to the x grid interval
    '''
    import numpy as N
    
    sum = N.sum
    if x != None:
        dx = (x[-1]-x[0])/(N.shape(x)[0]-1)
    if y != None:
        dy = (y[-1]-y[0])/(N.shape(y)[0]-1)    
    
    s1 = z[0,0] + z[-1,0] + z[0,-1] + z[-1,-1]
    s2 = sum(z[1:-1,0]) + sum(z[1:-1,-1]) + sum(z[0,1:-1]) + sum(z[-1,1:-1])
    s3 = sum(z[1:-1,1:-1])
    
    return 0.25*dx*dy*(s1 + 2*s2 + 4*s3)
################################################
import numba
from numba import cfunc
from numba import jit
@jit
def self_energy_mw( p_in, w_in ) :
 return scipy.integrate.tplquad( lambda  p, w, theta :
(-2.0/(scipy.pi)**2)*(im_rpa_pw(p,w)/( (im_rpa_pw(p,w))**2 + (re_rpa_pw(p,w))**2  ))*(1.0/( w_in - w - p_in**2 - p**2 + 2*p*p_in*scipy.cos(theta) ) ) * ( n_bose_forw(w) - (1.0/(scipy.exp(p_in**2 + p**2 - 2*p*p_in*scipy.cos(theta) - gamma)/Tm )  + 1.0 ) ) , -0.5*scipy.pi , 0.5*scipy.pi,  lambda theta : 0.2 , lambda theta : 0.25 , lambda theta , w: 0.1, lambda theta, w: 0.15  )[1]   
#Needs two arguments for calling and one dummy for integrating over angles


#############################################################################################
import numpy.random as rnd
def integrateMC(func, dim, limit, N=100):
    """
     sin(x*y) is taken as f(x)= sin(x[0]*x[1])
    """
    I =1.0/N
    sum = 0
    for n in range(dim):
        I *= (limit[n][1] - limit[n][0])
        
    for k in range(N):
        x = []
        for n in range(dim):
            x += [limit[n][0] + (limit[n][1] - limit[n][0])*rnd.random()]
        sum += func(x)
    return I*sum
#############################################################################################
""" to use any function for integration pass using wrapper function  """
""" Example : 1D Integral of imrpa(w) w.r.t w in some domain
def wrapper_imrpa(bb):
 return imrpa( bb[0]  )
integrateMC(  wrapper_imrpa  , 1, [[1,2]], N=1000) """

""" Example : 2D Integral of im_rpa_pw(w) w.r.t  w, p in some domain
def wrapper_im_rpa_pw(bb):
 return im_rpa_pw( bb[0], bb[1]  )
integrateMC(  wrapper_im_rpa_pw  , 2, [[0.2,0.4],[1,2]], N=1000) """
#############################################################################################    

def wrapper_montrollward_plot(w_input1, w_input2, n_wpoints, p_input  ):
 self_energy_montrollward_pwt = lambda  p, w, theta : (-2.0/(scipy.pi)**2)*(im_rpa_pw(p,w)/( (im_rpa_pw(p,w))**2 + (re_rpa_pw(p,w))**2  ))*(1.0/( w_in[i] - w - p_in**2 - p**2 + 2*p*p_in*scipy.cos(theta) ) ) * ( n_bose_forw(w) - (1.0/(scipy.exp(p_in**2 + p**2 - 2*p*p_in*scipy.cos(theta) - gamma)/Tm )  + 1.0 ) )   
 def wrapper_self_energy_montrollward_pwt(bb):
    return self_energy_montrollward_pwt(bb[0], bb[1], bb[2])   
    
 mwself_ans =[];
 w_in = numpy.linspace( w_input1, w_input2, n_wpoints  );
 p_in = p_input;
 N = mc_points;
 for i in range( len(w_in) ):
  mwself_ans.append( integrateMC( wrapper_self_energy_montrollward_pwt , 3, [ [0.0, 4.0 * k_f ],[ 0, 4.0 * w_pl ],[-0.5*scipy.pi,0.5*scipy.pi] ], N) )   
 plt.plot( w_in, mwself_ans  )   
#############################################################################################   
def wrapper_self_energy_im_forw_plot(w_input1, w_input2, n_wpoints, p_input, mc_points  ):
 self_energy_im_forw =  lambda p_dummy, w_dummy : ( (2.0 * scipy.pi)**-4  ) *  greensfn_forw( p_dummy , w_dummy  ) * W_forw( p_in - p_dummy, w_in[i] - w_dummy)
 def wrapper_self_energy_im_forw(bb):
    return self_energy_im_forw(bb[0], bb[1]) 
    
 mwself_ans2 =[];
 w_in = numpy.linspace( w_input1, w_input2, n_wpoints, mc_points  );
 p_in = p_input;
 N = mc_points;
 for i in range( len(w_in) ):
  mwself_ans2.append( integrateMC( wrapper_self_energy_im_forw , 2 , [ [0.0, 2.0 * k_f ],[ 0, 2.0 * w_pl ] ], N) )   
 plt.plot( w_in, mwself_ans2  )   
#############################################################################################   
def wrapper_self_energy_im_back_plot(w_input1, w_input2, n_wpoints, p_input , mc_points ):
 self_energy_im_back =  lambda p_dummy, w_dummy : ( (2.0 * scipy.pi)**-4  ) * greensfn_back( p_dummy , w_dummy  ) * W_back( p_in - p_dummy, w_in[i] - w_dummy)
 def wrapper_self_energy_im_back(bb):
    return self_energy_im_back(bb[0], bb[1]) 
    
 mwself_ans3 =[];
 w_in = numpy.linspace( w_input1, w_input2, n_wpoints  );
 p_in = p_input;
 N = mc_points;
 for i in range( len(w_in) ):
  mwself_ans3.append( integrateMC( wrapper_self_energy_im_back , 2 , [ [0.0, 2.0 * k_f ],[ 0, 2.0 * w_pl ] ], N) )   
 plt.plot( w_in, mwself_ans3  )   
############################################################################################# 
def structure_factor(w_input1, w_input2, n_wpoints, p_input):
  sf = lambda p, w : (1.0/scipy.pi)*(p**2 / (4.0 * scipy.pi) )* ( im_rpa_pw(p,w) /( im_rpa_pw(p,w) + re_rpa_pw(p,w)   ) ) * n_bose_forw(w)
  sf_ans = [];
  w_in = numpy.linspace( w_input1, w_input2, n_wpoints  );
  p_in = p_input;
  for i in range( len(w_in) ):
     sf_ans.append( sf( p_in, w_in[i] )   )
     
  plt.plot(w_in , sf_ans)
#############################################################################################    
def wrapper_self_energy_plot(w_input1, w_input2, n_wpoints, p_input, mc_points  ):
 self_energy_im_back =  lambda p_dummy, w_dummy : ( (2.0 * scipy.pi)**-4  ) * greensfn_back( p_dummy , w_dummy  ) * W_back( p_in - p_dummy, w_in[i] - w_dummy)
 def wrapper_self_energy_im_back(bb):
    return self_energy_im_back(bb[0], bb[1]) 
 self_energy_im_forw =  lambda p_dummy, w_dummy : ( (2.0 * scipy.pi)**-4  ) * greensfn_forw( p_dummy , w_dummy  ) * W_forw( p_in - p_dummy, w_in[i] - w_dummy)
 def wrapper_self_energy_im_forw(bb):
    return self_energy_im_forw(bb[0], bb[1])   
 self_energy_im = lambda p_dummy, w_dummy : ( (2.0 * scipy.pi)**-4  ) * ( greensfn_forw( p_dummy , w_dummy  ) * W_forw( p_in - p_dummy, w_in[i] - w_dummy)  -  greensfn_back( p_dummy , w_dummy  ) * W_back( p_in - p_dummy, w_in[i] - w_dummy) )
 def wrapper_self_energy_im(bb):  
    return self_energy_im(bb[0], bb[1])      
 mwself_ans4 =[];
 w_in = numpy.linspace( w_input1, w_input2, n_wpoints  );
 p_in = p_input;
 N = mc_points;
 for i in range( len(w_in) ):
  mwself_ans4.append( 0.5 * integrateMC(   wrapper_self_energy_im, 2 , [ [-4.0 * k_f, 4.0 * k_f ],[ -0.98 * w_pl, 0.98 * w_pl ] ], N) )   
 plt.plot( w_in, mwself_ans4 )  
#############################################################################################
""" Wrapper functions """
self_energy_im_back =  lambda p_dummy, w_dummy :  ( (2.0 * scipy.pi)**-4  ) * greensfn_back( p_dummy , w_dummy  ) * W_back( p_in - p_dummy, w_in - w_dummy)
self_energy_im_forw =  lambda p_dummy, w_dummy : lambda p_in, w_in: ( (2.0 * scipy.pi)**-4  ) * greensfn_forw( p_dummy , w_dummy  ) * W_forw( p_in - p_dummy, w_in - w_dummy)
self_energy_im = lambda p_dummy, w_dummy : lambda p_in, w_in: ( (2.0 * scipy.pi)**-4  ) * ( greensfn_forw( p_dummy , w_dummy  ) * W_forw( p_in - p_dummy, w_in - w_dummy)  -  greensfn_back( p_dummy , w_dummy  ) * W_back( p_in - p_dummy, w_in - w_dummy) )
def wrapper_self_energy_im_back(bb):
    return self_energy_im_back(bb[0], bb[1])
def wrapper_self_energy_im_forw(bb):
    return self_energy_im_forw(bb[0], bb[1])
def wrapper_self_energy_im(bb):  
    return self_energy_im(bb[0], bb[1])   
def montecarlo_int(wrapperfn,dim,xarray,yarray, N_points):
    return integrateMC(wrapperfn, dim, [ xarray, yarray ], N = N_points   )     
#############################################################################################
""" Wrapper functions as functions of momentum and frequency  given as dummy variables"""
""" Needs input of variables p_in, w_in before montecarlo integration """ 
""" Example of self-energy calculation after declaring p_in, w_in  """
""" 0.5 * integrateMC(   wrapper_self_energy_im, 2 , [ [0.0, 2.0 * k_f ],[ 0, 2.0 * w_pl ] ], 50 ) """    

""" real part of self-energy using Kramers-Kronig with Hilbert transform   """
""" hs1 = scipy.fftpack.hilbert( [array] ) """
self_energy_im_back_ex =  lambda p_dummy, w_dummy : lambda p_in, w_in : ( (2.0 * scipy.pi)**-4  ) * greensfn_back( p_dummy , w_dummy  ) * W_back( p_in - p_dummy, w_in - w_dummy)
""" Declare lambda inside lambda of the form  f = lambda x ,y :  lambda x1,y1 : x * (y**2) * (x1**3) * (y1**4) """
""" Call function using f(x,y)(x1,y1)   """
""" 1d Integrate of any variable using scipy.integrate.quad( lambda x : f(1,1)(x,1) , 0,1 ) """
""" store self-energy imaginary values in an array for corresponding p, w values """
class selfenergy :
 def _init_(self,a,b):
  self.a = a
  self.b = b
 @staticmethod 
 def n_bose_back(w):
  return n_bose_back(w)
 @staticmethod  
 def n_bose_forw(w):
  return n_bose_forw(w)
 @staticmethod 
 def n_fermi_forw(p):
  return  1.0 / ( scipy.exp( ( (p**2) - gamma)/Tm )   + 1.0) 
 @staticmethod
 def n_fermi_back(p):
  return  1.0 - (1.0 / ( scipy.exp( ( (p**2) - gamma)/Tm )   + 1.0) )
  
  
#############################################################################################
""" Define iterator 1 for GW   """
def spectral_fn_iter(  p ,w  ):
 n_bose_back = lambda w: 1.0 / ( scipy.exp( (w - gamma)/Tm ) -  1.0) 
 n_bose_forw = lambda w: 1.0 + ( 1.0 / ( scipy.exp( (w - gamma)/Tm ) -  1.0) )
 n_fermi_forw = lambda p:  1.0 / ( scipy.exp( ( (p**2) - gamma)/Tm )   + 1.0) 
 n_fermi_back = lambda p:  1.0 - (1.0 / ( scipy.exp( ( (p**2) - gamma)/Tm )   + 1.0) ) 
 greensfn_forw = lambda p, w : spec_fn_p_w(p,w) * n_fermi_forw(p)
 greensfn_back =  lambda p, w : spec_fn_p_w(p,w) * n_fermi_back(p)
 V_non_int = lambda p : 4.0 *scipy.pi /(p**2) 
 W_forw = lambda p, w :  V_non_int(p)* (1.0 + n_bose_forw(w)) * (im_rpa_pw(p,w)/( (im_rpa_pw(p,w))**2 + (re_rpa_pw(p,w))**2  ))
 W_back =  lambda p, w : V_non_int(p)* n_bose_back(w) * (im_rpa_pw(p,w)/( (im_rpa_pw(p,w))**2 + (re_rpa_pw(p,w))**2  ))
 spec_fn_p_w = lambda  p, w : ( ( (2.0 * scipy.pi)**0.5 ) /( c_p(p) ) ) * scipy.exp( ( -( w - (p**2) - hartreefock(p)  )**2 )/ (2.0 * (c_p(p)**2) ) ) 
 spec_fn_norm = lambda p :  scipy.integrate.quad( lambda w : (0.5/scipy.pi)*( ( (2.0 * scipy.pi)**0.5 ) /( c_p(p) ) ) * scipy.exp( ( -( w - (p**2) - hartreefock(p)  )**2 )/ (2.0 * (c_p(p)**2) ) ) , -numpy.inf, numpy.inf )[0]
 c_p = lambda p : 3*(p**0)   
 sp_fn_moment1 = lambda p :  scipy.integrate.quad( lambda w: (w)*(0.5/scipy.pi)*(((2.0 * scipy.pi)**0.5)/( c_p(p) ) )*scipy.exp(( -( w - p**2 - hartreefock(p)  )**2 )/ (2.0 *c_p(p)**2)) , -numpy.inf, numpy.inf  )[0]
 sp_fn_moment2 = lambda p :  scipy.integrate.quad( lambda w: (w**2)*(0.5/scipy.pi)*(((2.0 * scipy.pi)**0.5)/( c_p(p) ) )*scipy.exp(( -( w - p**2 - hartreefock(p)  )**2 )/ (2.0 *c_p(p)**2)) , -numpy.inf, numpy.inf  )[0]
 sp_fn_moment3 = lambda p :  scipy.integrate.quad( lambda w: (w**3)*(0.5/scipy.pi)*(((2.0 * scipy.pi)**0.5)/( c_p(p) ) )*scipy.exp(( -( w - p**2 - hartreefock(p)  )**2 )/ (2.0 *c_p(p)**2)) , -numpy.inf, numpy.inf  )[0]
 def hartree_fock_self_energy(p1,p2,np):
  q_v = numpy.linspace(p1,p2,np);
  hfval = map( hartreefock, q_v  );
 """ Define function for self energy corrections """  
 spec_fn_with_self_energy = lambda p ,w : spec_fn_p_w(p, w - self_energy_re ) 
 spec_fn_with_self_energy_moment1 = lambda p : integrateMC( self, 10**1, 10**1, N=500  )
 spec_fn_with_self_energy_moment2 = lambda p : integrateMC( self, 10**1, 10**1, N=500  )
 """ Spectral function moments convergence  """ 
 E_1st_moment = lambda p: (p**2) + hartreefock(p)
 E_2nd_moment = lambda p: ( (p**2) + hartreefock(p) )**2 + integrateMC( wrapper_self_energy_im , 10**1, 10**1, N=500  )
   
 #self_energy_im_forw(p, w - self_energy_im )
 #self_energy_im_back(p, w - self_energy_im )
 def wrapper_self_energy(bb):
     return  self_energy_im_forw(bb[0] , bb[1]) -  self_energy_im_back( bb[0], bb[1] )
     
 def wrapper(bb):  
     return self_energy_im_back_ex(bb[0], bb[1])(p_in,w_in)    
#############################################################################################
""" Vegas integrator 1d functions""" 

#def f(x):
#    dx2 = 0
#    for d in range(1):
#        dx2 += ( (x[d])**2  + 2.0 ) 
#    f = dx2    
#    return f

#def f(x):
#    return x[0] ** 2
#
#integ = vegas.Integrator([[0, 1]])
#
#result = integ(f, nitn=1000, neval=1000)
#print(result.summary())
#print('result = %s    Q = %.2f' % (result, result.Q))
############################################################################################# 
""" Vegas integrator 2d functions"""
#def f(x):
#    return x[0] ** 2 + x[1] ** 3    
#    
#integ2 = vegas.Integrator([[0, 1],[0,1]])
#
#result2 = integ2(f, nitn=1000, neval=1000)
############################################################################################# 
""" Vegas for imaginary self-energy part """
#def f(x):
#   return ( (2.0 * scipy.pi)**-4  ) * ( greensfn_forw( x[0] , x[1]  ) * W_forw( p_in - x[0], w_in[i] - x[1] )  -  greensfn_back( x[0] , x[1]  ) * W_back( p_in - x[0], w_in[i] -  x[1] ) )
#    
#   
#integ2se = vegas.Integrator([[0, 2.0 * w_pl ],[0, 2.0 * k_f ]])
#
#result2se = integ2se(f, nitn=20, neval=20)
#
#answ = []
#for i in w_in:
#    result2se = integ2se(f, nitn=10, neval=10)
#    answ.append( 0.5 * result2se )
#############################################################################################
"""  Declaring function for vegas """
""" Example to pass some random function fr as ff for vegas"""
""" def fr(p, w) :
     return ( p - p_in  ) * ( w - w_in  )**2 """
""" def ff(x) :
     return fr( x[0] , x[1]  ) """
#############################################################################################

self_energy_montrollward_pwt = lambda  p, w, theta : (-2.0/(scipy.pi)**2)*(im_rpa_pw(p,w)/( (im_rpa_pw(p,w))**2 + (re_rpa_pw(p,w))**2  ))*(1.0/( w_in[i] - w - p_in**2 - p**2 + 2*p*p_in*scipy.cos(theta) ) ) * ( n_bose_forw(w) - (1.0/(scipy.exp(p_in**2 + p**2 - 2*p*p_in*scipy.cos(theta) - gamma)/Tm )  + 1.0 ) )   

def mw_vegas(x) :
    return self_energy_montrollward_pwt( x[0], x[1], x[2]  )
    
integ = vegas.Integrator( [ [ 0, 4.0  * k_f  ], [ -0.99 * w_pl, 0.99 * w_pl   ], [-0.5*scipy.pi,0.5*scipy.pi] , ]  )

p_in = 1.0;

answ = [];
w_in = numpy.linspace(-4,4,5)
#for i in w_in:
#   result2se = integ(mw_vegas, nitn=10, neval=1000)
#   answ.append( result2se.val )
#   
   
  
