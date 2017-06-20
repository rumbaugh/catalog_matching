import os
import sys
import numpy as np
import time

def SphDist(RAi,Deci,Raf,Decf):
        #Computes distance in arcminutes on the sky using spherical geometry
        #Inputs:
        #     Rai - Right Ascension of first object(s) in decimal degrees
        #     Deci - Declination of first object(s) in decimal degrees
        #     Raf - Right Ascension of second object(s) in decimal degrees
        #     Deci - Declination of second object(s) in decimal degrees
        #Outputs:
        #     dist - distance, or array of distances, in arcminutes
        # Both the first and second set of inputs may be arrays
        dist = 2.0*np.arcsin(np.sqrt((np.sin(0.5*np.pi*(Decf-Deci)/180))**2+np.cos(np.pi*Decf/180.0)*np.cos(np.pi*Deci/180.0)*(np.sin(0.5*np.pi*(Raf-RAi)/180))**2))
        dist *= 180.0*60/np.pi
        return dist

def FindCloseSources(ra,dec,tol,ra_opt,dec_opt,minsrch=0.0):
        #Locates sources in the reference catalog near a position (ra,dec).
        #Inputs:
        #     ra - Right Ascension of position to locate sources near, in decimal degrees
        #     dec - Declination of position to locate sources near, in decimal degrees
        #     tol - Distance from position, in arcseconds, in which to search
        #     ra_opt - Array of right ascensions, in decimal degrees, to search
        #     dec_opt - Array of declinations, in decimal degrees, to search
        #     minsrch - Minimum value for tol, in arcseconds
        #Outputs:
        #     ra_box_temp_ind1 - Array containing indices representing sources in the 
        #              reference catalog within the distance tol of the input position
        if ((tol < minsrch) & (minsrch>0)): tol = minsrch
        #First, find sources in a box with sides equal to the search radius. This saves
        #time when calculated the distance to the references sources
        tolx = tol/np.cos(dec*np.pi/180.0)
        ra_box_temp_ind1 = np.where((ra_opt >= ra-tolx/3600.0) & (ra_opt <= ra+tolx/3600.0) & (dec_opt >= dec-tol/3600.0) & (dec_opt <= dec+tol/3600.0))[0]
        if len(ra_box_temp_ind1) > 0: 
                #Now, calculate distances and select those within the search radius
                temp_dist_ar = SphDist(ra_opt[ra_box_temp_ind1],dec_opt[ra_box_temp_ind1],ra,dec)
                inside_tol_ind = np.where(temp_dist_ar*60.0 <= tol)[0]
                if len(inside_tol_ind) > 0: 
                        return ra_box_temp_ind1[inside_tol_ind]
                else:
                        return np.zeros(0)
        else:
                return np.zeros(0)
                
def OptMatch(raX,decX,d_xerr,ra_opt,dec_opt,loc_num_dens_gtMag,minsrch,takesqrt_numdens=True):
        #Performs matching for an input source to the given reference catalog
        #Inputs:
        #     raX - Right Ascension of source to be matched in decimal degrees
        #     decX - Declination of source to be matched in decimal degrees
        #     errX - Positional error of the input source in arcseconds
        #     ra_opt - Array of right ascensions, in decimal degrees, to search
        #     dec_opt - Array of declinations, in decimal degrees, to search
        #     loc_num_dens_gtMag - Number density of sources with magnitudes greater than the source magnitude, given for each source in the referenace catalog
        #     minsrch - Minimum value for search radius, in arcseconds
        #     takesqrt_numdens - If true, the square root of the number density is used in place of the number density for weighting the likelihoods
        #Outputs:
        #     close_likelihoods - Likelihoods for the match
        #     close_ind - Index representing sources in the reference catalog within the positional error
        close_ind = FindCloseSources(raX,decX,d_xerr,ra_opt,dec_opt,minsrch)
        if len(close_ind) > 0:
                ra_close,dec_close,numdens_close = ra_opt[close_ind],dec_opt[close_ind],loc_num_dens_gtMag[close_ind]
                close_dist=60*SphDist(raX,decX,ra_close,dec_close)
                #This is the formula for the likelihoods
                if takesqrt_numdens: numdens_close=np.sqrt(numdens_close) # change the weighting to square root of number density
                close_likelihoods = np.exp(-0.5*(close_dist**2)/(d_xerr**2))/(d_xerr**2.0*numdens_close)
                return close_likelihoods,close_ind
        else:   
                return np.zeros(0),np.zeros(0)
        
def MonteCarlo(reps,ra_rand,dec_rand,err,ra_opt,dec_opt,loc_num_dens_gtMag,minsrch,takesqrt_numdens=True):
        #Performs Monte Carlo simulation to calculate reliabilities for a given likelihood
        #Inputs:
        #     reps - Number of MC trials
        #     ra_rand - Array of random RAs for the Monte Carlo simulation, with length equal to reps
        #     dec_rand - Array of random Decs for the Monte Carlo simulation, with length equal to reps
        #     ra_opt - Array of right ascensions, in decimal degrees, to search
        #     dec_opt - Array of declinations, in decimal degrees, to search
        #     loc_num_dens_gtMag - Number density of sources with magnitudes greater than the source magnitude, given for each source in the referenace catalog
        #     minsrch - Minimum value for search radius, in arcseconds
        #     takesqrt_numdens - If true, the square root of the number density is used in place of the number density for weighting the likelihoods
        #Outputs:
        #     outputs - Array containing likelihoods from the Monte Carlo simulation
        output = np.zeros(0)
        for i in range(0L,reps):
                likelihoods,like_inds = OptMatch(ra_rand[i],dec_rand[i],err,ra_opt,dec_opt,loc_num_dens_gtMag,minsrch,takesqrt_numdens=takesqrt_numdens)
                output = np.append(output,likelihoods)
        return output

def FastMonteCarlo(reps,ra_rand,dec_rand,ra_opt,dec_opt,loc_num_dens_gtMag,inds,like_arr,minsrch,takesqrt_numdens=True,verbose=True):
        #Sources that use the minimum search radius have the results from the earlier MC run input here
        errMC=d_xerr[inds]
        maxsrch=np.max(errMC)
        #Actual Monte Carlo simulations start here. The same number (reps) of random positions are used for
        #every source in the input catalog, since the MC sims only depend on search radius.
        if verbose: print "Starting Monte Carlo simulations with %i trials..."%reps
        for j in np.arange(0,reps):
                if verbose:
                        if j == int(reps/4.0):
                                #print time.time(),start_time,setup_time
                                quarter_time = time.time()-start_time-setup_time-like_time
                                print "MC is 25%% done - ETA: %.1f seconds."%(3*quarter_time)
                        elif j == int(reps/2.0):
                                #print time.time(),start_time,setup_time
                                half_time = time.time()-start_time-setup_time-like_time
                                print "MC is 50%% done - ETA: %.1f seconds."%(half_time)
                        elif j == int(3*reps/4.0):
                                #print time.time(),start_time,setup_time
                                quarter_time = time.time()-start_time-setup_time-like_time
                                print "MC is 75%% done - ETA: %.1f seconds." %(quarter_time/3.0)
                #First, find all sources within the maximum search radius (of all input sources). 
                #This saves times because we need to calculate distances for all of these.
                tolx = maxsrch/np.cos(dec_rand[j]*np.pi/180.0)
                ra_box_temp_ind1 = np.where((ra_opt >= ra_rand[j]-tolx/3600.0) & (ra_opt <= ra_rand[j]+tolx/3600.0) & (dec_opt >= dec_rand[j]-maxsrch/3600.0) & (dec_opt <= dec_rand[j]+maxsrch/3600.0))[0]
                #Okay, now calculate distances
                temp_dist_ar = SphDist(ra_opt[ra_box_temp_ind1],dec_opt[ra_box_temp_ind1],ra_rand[j],dec_rand[j])*60
                #Which ones are within the max search radius?
                inside_maxtol = np.where(temp_dist_ar <= maxsrch)[0]
                mcinds=np.arange(0,len(inds))#This index means we don't redo MC sims where the minimum search radius is used
                for i in mcinds:
                        #Find sources within the search radius for source i
                        if minsrch>errMC[i]:
                                sph_ind_tmp=inside_maxtol[temp_dist_ar[inside_maxtol]<minsrch]
                        else:
                                sph_ind_tmp=inside_maxtol[temp_dist_ar[inside_maxtol]<errMC[i]]
                        close_ind=ra_box_temp_ind1[sph_ind_tmp]
                        if len(close_ind) > 0:
                                numdens_close = loc_num_dens_gtMag[close_ind]
                                close_dist=temp_dist_ar[sph_ind_tmp] #Distances were already calculated, so we just need to tack the index on
                                if takesqrt_numdens: numdens_close=np.sqrt(numdens_close) # change the weighting to square root of number density
                                close_likelihoods = np.exp(-0.5*(close_dist**2)/(errMC[i]**2))/(errMC[i]**2.0*numdens_close)
                        else:
                                close_likelihoods=np.zeros(0)
                        #Append the results of this MC trial to the array of all MC trial likelihoods for that object
                        like_arr[inds[i]]=np.append(like_arr[inds[i]],close_likelihoods)
        return like_arr

def Setup_Xray(cat_in,load_dict={'names':('raX','decX','fluxX_soft','fluxX_hard','fluxX_full','netcnts_corrX_soft','netcnts_corrX_hard','netcnts_corrX_full','sigX_soft','sigX_hard','sigX_full','dummy1','dummy2','dummy3','wflagX'),'formats':('f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','i4')},minerr=1.5):
        #Sets up MatchCat function for X-ray matching
        #Inputs:
        #     cat_in - Path to input X-ray catalog
        #     load_dict - dictionary for loading cat_in, with names and formats of columns
        #This sets ups the parameters raX, decX, sigmax, xwnetcnts, and d_xerr for later use in MatchCat.
        if load_dict==None: load_dict={'names':('raX','decX','fluxX_soft','fluxX_hard','fluxX_full','netcnts_corrX_soft','netcnts_corrX_hard','netcnts_corrX_full','sigX_soft','sigX_hard','sigX_full','dummy1','dummy2','dummy3','wflagX'),'formats':('f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','i4')}
        global raX,decX,sigmax,xwnetcnts,d_xerr
        cr_xray = np.loadtxt(cat_in,dtype=load_dict)
        raX,decX,netcnts_corrX_soft,netcnts_corrX_hard,netcnts_corrX_full=cr_xray['raX'],cr_xray['decX'],cr_xray['netcnts_corrX_soft'],cr_xray['netcnts_corrX_hard'],cr_xray['netcnts_corrX_full']
        #find max sigma for each source out of all bands
        if usesigflag:
                sigX_soft,sigX_hard,sigX_full=cr_xray['sigX_soft'],cr_xray['sigX_hard'],cr_xray['sigX_full']
                sigma_matrix = np.concatenate((np.reshape(sigX_soft,(len(sigX_soft),1)),np.reshape(sigX_hard,(len(sigX_hard),1)),np.reshape(sigX_full,(len(sigX_full),1))),axis=1)
                sigmax = np.max(sigma_matrix,axis=1)
        else:
                sigmax = np.ones(len(raX))*4.
        try:
                wflagX=cr_xray['wflagX']
                tmptest=np.zeros(3)[wflagX]
        except:
                if usesigflag:
                        wflagX=np.argsort(sigma_matrix)[:,-1]
                else:
                        ncnts_matrix=np.concatenate((np.reshape(netcnts_corrX_soft,(len(netcnts_corrX_soft),1)),np.reshape(netcnts_corrX_hard,(len(netcnts_corrX_hard),1)),np.reshape(netcnts_corrX_full,(len(netcnts_corrX_full),1))),axis=1)
                        wflagX=np.argsort(ncnts_matrix)[:,-1]
        #consolidate net counts in one matrix (soft,hard,full is columns 0,1,2)
        ncnts_corrX = np.concatenate((np.reshape(netcnts_corrX_soft,(len(netcnts_corrX_soft),1)),np.reshape(netcnts_corrX_hard,(len(netcnts_corrX_hard),1)),np.reshape(netcnts_corrX_full,(len(netcnts_corrX_full),1))),axis=1)
        #set off axis angles
        if np.shape(ra_opt_cen)!=():
                off_axisX_tmp=np.zeros((len(raX),len(ra_opt_cen)))
                for ioa in range(0,len(ra_opt_cen)): off_axisX_tmp[:,ioa]=SphDist(ra_opt_cen[ioa],dec_opt_cen[ioa],raX,decX)
                off_axisX=np.min(off_axisX_tmp,axis=1)
        else:
                off_axisX = SphDist(ra_opt_cen,dec_opt_cen,raX,decX)
        #for jk in range(0,len(off_axisX)): print raX[jk],decX[jk],off_axisX[jk],off_axisX_tmp[jk]
        xwnetcnts = ncnts_corrX[(np.arange(len(wflagX)),wflagX)]

        # Calculate errors
        d_xerr = np.zeros(len(raX))
        d_xerr[xwnetcnts<=137.816] = 10**(0.1145*off_axisX[xwnetcnts<=137.816] - 0.4958*np.log10(xwnetcnts[xwnetcnts<=137.816])+0.1932)
        d_xerr[xwnetcnts>137.816] = 10**(((0.0968*off_axisX[xwnetcnts>137.816] - 0.2064*np.log10(xwnetcnts[xwnetcnts>137.816])-0.4260)))
        d_xerr[off_axisX>=15.] = 60.*np.ones(len(off_axisX[off_axisX>=15.]))
        d_xerr[xwnetcnts<=0] = 60.*np.ones(len(xwnetcnts[xwnetcnts<=0]))
        if minerr!=None:d_xerr[d_xerr<minerr] = minerr*np.ones(len(d_xerr[d_xerr<minerr]))
        return

def Setup_Radio(cat_in,load_dict={'names':('ID','raX','decX','dRA','dDec','PkFit','dPk','IntFit','dInt'),'formats':('i8','f8','f8','f8','f8','f8','f8','f8','f8')},minerr=None):
        #Sets up MatchCat function for radio matching
        #Inputs:
        #     cat_in - Path to input radio catalog
        #     load_dict - dictionary for loading cat_in, with names and formats of columns
        #This sets ups the parameters raX, decX, radioflux, and d_xerr for later use in MatchCat.
        if load_dict==None:load_dict={'names':('ID','raX','decX','dRA','dDec','PkFit','dPk','IntFit','dInt'),'formats':('i8','f8','f8','f8','f8','f8','f8','f8','f8')}
        global raX,decX,radioflux,d_xerr
        cr_radio=np.loadtxt(cat_in,dtype=load_dict)
        raX,decX,dRA,dDec,PkFit,IntFit=cr_radio['raX'],cr_radio['decX'],cr_radio['dRA'],cr_radio['dDec'],cr_radio['PkFit'],cr_radio['IntFit']
        gddec=np.where(dDec>dRA)[0]
        d_xerr=np.copy(dRA)
        d_xerr[gddec]=dDec[gddec]
        gIntFit=np.where(IntFit>PkFit)
        radioflux=np.copy(PkFit)
        radioflux[gIntFit]=IntFit[gIntFit]
        if minerr!=None:d_xerr[d_xerr<minerr] = minerr*np.ones(len(d_xerr[d_xerr<minerr]))
        return


def MatchCat(cat_in,ref_cat,outcat,outreg,Ra_aim=None,Dec_aim=None,setup='radio',minsrch=5.0,minerr=1.5,reps=10000,verbose=True,radius_opt_cen=0.2*3600,prob_thresh=0.15,load_dict=None,ref_load_dict={'names':('ra','dec','magI','ID'),'formats':('f8','f8','f8','|S64')},useConcaveHull=False,repsSecondPass=0,usesig=True,takesqrt_numdens=True):
        '''Match sources in an input catalog to a reference catalog. A likelihood and
        #reliability (probability of match) are calculated for each match, as well
        #as a probability that there is no true match. The ID, position, and 
        #positional error are output for each input source, as well as the IDs,
        #positions, and probabilities from the matching for the top three matches
        #for each source, as well as the probability of no match. Several parameters
        #specific to the setup will be written out, as well. A ds9 region file
        #is also created that shows the positional error of each matched input
        #source and the positions of the matched source(s) from the reference
        #catalog. 
        #
        #Inputs:
        #     cat_in - Path to catalog to be matched
        #     ref_cat - Path to reference catalog (to be matched to)
        #     outcat - Path to output data table file
        #     outreg - Path to output ds9 regions file for matches
        #     Ra_aim - Only those reference sources within radius_opt_cen of this RA will be matched to (in decimal degrees)
        #     Dec_aim - Only those reference sources within radius_opt_cen of this Dec will be matched to (in decimal degrees)
        #     setup - Defines which setup function will be called. Can be 'radio' or 'xray', or a custom function can be given
        #     minsrch - Minimum radius around each source in which to search for sources in the reference catalog (in arcseconds)
        #     minerr - Minimum positional error to use. During initial setup, all errors less than this value are set equal to minerr. Measured in arcseconds.
        #     reps - Number of Monte Carlo trials to use
        #     verbose - If True, prints out progress
        #     radius_opt_cen - Only those reference sources within radius_opt_cen of (Ra_aim,Dec_aim) will be matched to (in arcseconds)
        #     prob_thresh - Matches will not be accepted if the probability of no match is higher than this value
        #     load_dict - dictionary for loading cat_in, with names and formats of columns. None uses the default for the respective setup function
        #     ref_load_dict - dictionary for loading ref_cat, with names and formats of columns.
        #     takesqrt_numdens - If true, the square root of the number density is used in place of the number density for weighting the likelihoods'''

        global usesigflag
        usesigflag=usesig
        global start_time
        start_time = time.time()
        
        if minerr>minsrch:
                print "minerr is greater than minsrch parameter. Setting minsrch equal to minerr"
                minsrch=minerr

        #Load optical catalog
        cr_opt = np.loadtxt(ref_cat,dtype=ref_load_dict)
        ra_opt,dec_opt,magI_opt = cr_opt['ra'],cr_opt['dec'],cr_opt['magI']
        try:
                id_opt=cr_opt['ID']
        except ValueError:
                id_opt=np.arange(len(ra_opt))

        area_opt_cen = np.pi*(radius_opt_cen**2)
        global ra_opt_cen, dec_opt_cen
        if np.shape(Ra_aim)==():
                if Ra_aim!=None:
                        ra_opt_cen=Ra_aim
        else:
                ra_opt_cen=Ra_aim
        if np.shape(Dec_aim)==():
                if Dec_aim!=None:
                        dec_opt_cen=Dec_aim
        else:
                dec_opt_cen=Dec_aim
                


        # Run setup
        if isinstance(setup,basestring):
                if ((setup.lower()=='xray')|(setup.lower()=='x-ray')):setup=Setup_Xray
                elif setup.lower()=='radio': setup=Setup_Radio
        setup(cat_in,load_dict,minerr=minerr)

        ra_UB,ra_LB,dec_UB,dec_LB = np.max(raX),np.min(raX),np.max(decX),np.min(decX)

        if np.shape(Ra_aim)==():
                if Ra_aim==None:
                        ra_opt_cen=0.5*(ra_UB+ra_LB)
                if Dec_aim==None:
                        dec_opt_cen=0.5*(dec_UB+dec_LB)


        if useConcaveHull:
                from ConcaveHull import ConcaveHull,CheckPoints
                from shapely.geometry import box as makebox
                alpha_ref,alphaX = 100.1,1.1
                CH_inp_ref,CH_inpX = np.zeros((len(ra_opt),2)),np.zeros((len(raX),2))
                CH_inp_ref[:,0],CH_inp_ref[:,1],CH_inpX[:,0],CH_inpX[:,1]=ra_opt,dec_opt,raX,decX
                CHull_ref,edges_ref=ConcaveHull(CH_inp_ref,alpha_ref)
                CHullX,edgesX=ConcaveHull(CH_inpX,alphaX)
                testbox=makebox(ra_LB,dec_LB,ra_UB,dec_UB)
                if not(CHull_ref.contains(testbox)): #If this is true, then the simple box method is equivalent and there is no need to redo it
                        if verbose: print "Initial Concave Hull calculated. Time elapsed: %.1f seconds."%(time.time()-start_time)
                        CHull_both=CHull_ref.intersection(CHullX)
                else:
                        CHull_both=testbox
                area_opt_cen=CHull_both.area*3600**2
                

        #Get source number densities based on optical magnitudes. These are used to weight matches.
        if useConcaveHull:
                gmag=CheckPoints(CHull_both,ra_opt,dec_opt)
                magI_loc=magI_opt[gmag]
        else:
                d_fromoptcen = 60*SphDist(ra_opt,dec_opt,ra_opt_cen,dec_opt_cen)
                magI_loc = magI_opt[d_fromoptcen <= radius_opt_cen]
        
        loc_num_dens_gtMag = np.zeros(len(ra_opt))
        for i in range(0L,len(ra_opt)):
                ind_lemagI = np.where(magI_loc <= magI_opt[i])[0]
                num_gt_limit = len(ind_lemagI)
                loc_num_dens_gtMag[i] = num_gt_limit/area_opt_cen
        loc_num_dens_gtMag[loc_num_dens_gtMag==0]=0
        #Optical Matching

        #Initiate variables for matching
        matched_raX,matched_decX,matched_errX,matched_indX,matched_num_matches,matched_dec_opt1,matched_dec_opt2,matched_dec_opt3,matched_ra_opt1,matched_ra_opt2,matched_ra_opt3,matched_ind_opt1,matched_ind_opt2,matched_ind_opt3,matched_prob_opt1,matched_prob_opt2,matched_prob_opt3,matched_like_opt1,matched_like_opt2,matched_like_opt3 = -1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX)),-1*np.ones(len(raX))
        matched_prob_none = np.zeros(len(raX))
        
        #Set up variables for Monte Carlo simulation

        #If useConcaveHull is False, then the random points for the MC simulation
        #are chosen from within a box defined as below. This only works if that
        #box is itself enclosed by the reference catalog. If not, you need to use
        #the Concave Hull method
        ra_rand,dec_rand=np.random.uniform(ra_LB,ra_UB,reps),np.random.uniform(dec_LB,dec_UB,reps)
        if useConcaveHull:
                if verbose: print "Starting Concave Hull algorithm for MC setup, %.1f seconds elapsed."%(time.time()-start_time)
                #Uses a Concave Hull algorithm to determine the regions that define
                #the input catalogs and their intersection, then only generates 
                #random points within that region
                #from shapely.geometry import box as makebox
                #alpha_ref,alphaX = 100.1,1.1
                #CH_inp_ref,CH_inpX = np.zeros((len(ra_opt),2)),np.zeros((len(raX),2))
                #CH_inp_ref[:,0],CH_inp_ref[:,1],CH_inpX[:,0],CH_inpX[:,1]=ra_opt,dec_opt,raX,decX
                #CHull_ref,edges_ref=ConcaveHull(CH_inp_ref,alpha_ref)
                #CHullX,edgesX=ConcaveHull(CH_inpX,alphaX)
                #testbox=makebox(ra_LB,dec_LB,ra_UB,dec_UB)
                if not(CHull_ref.contains(testbox)): #If this is true, then the simple box method is equivalent and there is no need to redo it
                        if verbose: print "Starting random generation of points..."#%(time.time()-start_time)
                        from ConcaveHull import CHullRandomPoint
                        CHull_both=CHull_ref.intersection(CHullX)
                        ra_rand,dec_rand=CHullRandomPoint(CHull_both,reps)
                

        #Run Monte Carlo simulation for the minimum error. The Monte Carlo
        #simulation only depends on the search radius used, so this value can be
        #reused for all sources that use the minimum error.
        if minerr>0:
                if verbose: print "Doing MC simulation for minimum error, %.1f seconds elapsed"%(time.time()-start_time)
                min_ref = MonteCarlo(reps,ra_rand,dec_rand,minerr,ra_opt,dec_opt,loc_num_dens_gtMag,minsrch)
        like_dict={}
        global setup_time
        setup_time = time.time()-start_time
        if verbose: print "Setup took %.1f seconds."%(setup_time)

        #Calculate likelihoods for each input source, and store them in like_dict for later use
        global len_like_arr
        len_like_arr=np.zeros(len(raX))
        for i in range(0,len(raX)):
                if setup==Setup_Xray:
                        #For X-ray matching, sources with zero net counts are not matched
                        if xwnetcnts[i] > 0: 
                                likelihoods,like_inds = OptMatch(raX[i],decX[i],d_xerr[i],ra_opt,dec_opt,loc_num_dens_gtMag,minsrch,takesqrt_numdens=takesqrt_numdens)
                                len_like_arr[i]=len(likelihoods)
                        else:
                                likelihoods,like_inds = np.zeros(0),np.zeros(0)
                else:
                        likelihoods,like_inds = OptMatch(raX[i],decX[i],d_xerr[i],ra_opt,dec_opt,loc_num_dens_gtMag,minsrch,takesqrt_numdens=takesqrt_numdens)
                        len_like_arr[i]=len(likelihoods)
                like_dict[i]={'likelihoods':likelihoods,'like_inds':like_inds}
        global like_time
        like_time=time.time()-start_time-setup_time
        if verbose: print "Initial likelihood calculation complete. Took %.2f seconds."%like_time

        #Monte Carlo simulations
        
        #like_arr will store likelihood values from the MC sim. It is an object-type array,
        #so each entry will be an array (not all of the same size) storing all the 
        #likelihoods from the MC sim for that source
        like_arr=np.zeros(len(raX),dtype='object')
        for x in np.arange(0,len(raX)):
                if d_xerr[x]<=minerr:
                        like_arr[x]=min_ref
                else:
                        like_arr[x]=np.zeros(0)
        gDoFastMonteCarlo=np.arange(len(raX))[((d_xerr>minerr)&(len_like_arr>0))]
        if len(gDoFastMonteCarlo)>0:like_arr=FastMonteCarlo(reps,ra_rand,dec_rand,ra_opt,dec_opt,loc_num_dens_gtMag,gDoFastMonteCarlo,like_arr,minsrch,takesqrt_numdens,verbose=verbose)


        if verbose:
                mctime=time.time()-start_time-setup_time
                print "Monte Carlo simulations complete. Took %.2f seconds"%(mctime)

        if repsSecondPass>reps:
                if verbose: print "Starting follow-up Monte Carlo simulations..."
                g2ndpass=np.zeros(0,dtype='int')
                for i in range(0,len(raX)):
                        #Load the likelihoods for each source
                        likelihoods,like_inds=like_dict[i]['likelihoods'],like_dict[i]['like_inds']
                        num_matches = len(likelihoods)
                        if num_matches > 0:
                                rel_ref = np.copy(like_arr[i])
                                if setup==Setup_Xray:
                                        #No matching is done for X-ray sources with no net counts
                                        if xwnetcnts[i] <= 0: rel_ref = np.zeros(0) 
                        else:
                                rel_ref = np.zeros(0)
                        relflag=False
                        for j in range(0,num_matches):
                                #The number of MC trials with likelihoods less than actual likelihood
                                #calculated determines the reliability for that source
                                MC_temp = np.where(rel_ref >= likelihoods[j])[0]
                                rel_cnt = len(MC_temp)
                                if rel_cnt==0:
                                        relflag=True
                        if relflag:g2ndpass=np.append(g2ndpass,i)
                                        #break
                if len(g2ndpass)>0:
                        ra_UB,ra_LB,dec_UB,dec_LB = np.max(raX),np.min(raX),np.max(decX),np.min(decX)
                        ra_rand,dec_rand=np.random.uniform(ra_LB,ra_UB,repsSecondPass-reps),np.random.uniform(dec_LB,dec_UB,repsSecondPass-reps)
                        if ((useConcaveHull)&(not(CHull_ref.contains(testbox)))):
                                if verbose: 
                                        beginR_time=time.time()
                                        print "Creating new set of random points..."
                                ra_rand,dec_rand=CHullRandomPoint(CHull_both,repsSecondPass-reps)
                                if verbose: print "Random points generated. Took %.1f seconds"%(time.time()-beginR_time)
                        like_arr=FastMonteCarlo(repsSecondPass-reps,ra_rand,dec_rand,ra_opt,dec_opt,loc_num_dens_gtMag,g2ndpass,like_arr,minsrch,takesqrt_numdens,verbose=verbose)
                
                if verbose: print "Follow-up Monte Carlo simulations complete. Took %.2f seconds"%(time.time()-start_time-mctime-setup_time-like_time)
                
        reps=np.ones(len(raX))*reps
        if repsSecondPass>reps[0]: reps[g2ndpass]=repsSecondPass
        #Now we use the output of the MC sims to calculate probabilities for the matches
        for i in range(0,len(raX)):
                #Load the likelihoods for each source
                likelihoods,like_inds=like_dict[i]['likelihoods'],like_dict[i]['like_inds']
                num_matches = len(likelihoods)
                reliability_ij = np.zeros(num_matches)
                if num_matches > 0:
                        rel_ref = np.copy(like_arr[i])
                        if setup==Setup_Xray:
                                #No matching is done for X-ray sources with no net counts
                                if xwnetcnts[i] <= 0: rel_ref = np.zeros(0) 
                else:
                        rel_ref = np.zeros(0)
                for j in range(0,num_matches):
                        #The number of MC trials with likelihoods less than actual likelihood
                        #calculated determines the reliability for that source
                        MC_temp = np.where(rel_ref >= likelihoods[j])[0]
                        rel_cnt = len(MC_temp)
                        reliability_ij[j] = 1.0 - (rel_cnt*1.0)/reps[i]
                reliability_ij[reliability_ij< 0] = 0.0
                probability_ij = np.zeros(num_matches)
                for j in range(0,num_matches):
                        probability_ij[j] = reliability_ij[j]
                        for k in range(0,num_matches):
                                if k != j: probability_ij[j] *= (1-reliability_ij[k])
                #Determine the probability of no match
                probability_nonej = np.prod(1-reliability_ij)
                prob_sum = probability_nonej + np.sum(probability_ij)
                if prob_sum >= 0: 
                        probability_ij /= prob_sum
                        probability_nonej /= prob_sum
                else:
                        #If all the matches had a probability of zero, then there is a 100% chance of no match
                        probability_nonej = 1.0
                #Set matched parameters
                matched_raX[i],matched_decX[i],matched_errX[i],matched_indX[i],matched_prob_none[i] = raX[i],decX[i],d_xerr[i],i,probability_nonej
                if num_matches == 1:
                        matched_ra_opt1[i],matched_dec_opt1[i],matched_prob_opt1[i],matched_ind_opt1[i],matched_like_opt1[i] = ra_opt[like_inds[0]],dec_opt[like_inds[0]],1-probability_nonej,like_inds[0],likelihoods[0]
                        if probability_nonej <= prob_thresh: 
                                matched_num_matches[i] = 1
                        else:
                                matched_num_matches[i] = 0
                if num_matches > 1:
                        #If there is more than 1 match, we need to select out the three
                        #most probable matches and order them in order of probability
                        opt_matched_temp_ind = np.argsort(-1*likelihoods)
                        q = 0
                        opt_matched_ra_temp = np.array([-1.0,-1.0,-1.0])
                        opt_matched_dec_temp = np.array([-1.0,-1.0,-1.0])
                        opt_matched_prob_temp = np.array([-1.0,-1.0,-1.0])
                        opt_matched_ind_temp = np.array([-1.0,-1.0,-1.0])
                        opt_matched_like_temp = np.array([-1.0,-1.0,-1.0])
                        while ((q < 3) and (q < len(opt_matched_temp_ind))):
                                opt_matched_ra_temp[q] = ra_opt[like_inds[opt_matched_temp_ind[q]]]
                                opt_matched_dec_temp[q] = dec_opt[like_inds[opt_matched_temp_ind[q]]]
                                opt_matched_ind_temp[q] = like_inds[opt_matched_temp_ind[q]]
                                opt_matched_prob_temp[q] = probability_ij[opt_matched_temp_ind[q]]
                                opt_matched_like_temp[q] = likelihoods[opt_matched_temp_ind[q]]
                                q += 1
                        matched_ra_opt1[i],matched_dec_opt1[i],matched_prob_opt1[i], matched_ind_opt1[i],matched_like_opt1[i],matched_ra_opt2[i],matched_dec_opt2[i],matched_prob_opt2[i],matched_ind_opt2[i],matched_like_opt2[i],matched_ra_opt3[i], matched_dec_opt3[i],matched_prob_opt3[i],matched_ind_opt3[i],matched_like_opt3[i] = opt_matched_ra_temp[0],opt_matched_dec_temp[0], opt_matched_prob_temp[0],opt_matched_ind_temp[0],opt_matched_like_temp[0],opt_matched_ra_temp[1], opt_matched_dec_temp[1], opt_matched_prob_temp[1],opt_matched_ind_temp[1],opt_matched_like_temp[1],opt_matched_ra_temp[2],opt_matched_dec_temp[2], opt_matched_prob_temp[2],opt_matched_ind_temp[2],opt_matched_like_temp[2]
                        if probability_nonej <= prob_thresh:
                                #These two lines are a formula for selecting the matched sources 
                                #to use when there are more than one
                                prob_probe_temp = probability_ij/(1 - probability_nonej - probability_ij)
                                cand_temp_ind = np.where(prob_probe_temp >= 4.0)[0]
                                #Check consistency of results
                                if len(cand_temp_ind) > 1: sys.exit("Probabilities don't sum to 1")
                                if len(cand_temp_ind) == 1:
                                        matched_num_matches[i] = 1
                                if len(cand_temp_ind) == 0:
                                        if num_matches < 1: sys.exit("Inconsistency in matching...")
                                        test_sig_cands_ind = np.where(probability_ij >= 0.2)[0]
                                        if len(test_sig_cands_ind) > 3: 
                                                print "More than 3 matches for source %i. Truncating..."%i
                                                matched_num_matches[i] = 3
                                        elif len(test_sig_cands_ind) > 0:
                                                matched_num_matches[i] = len(test_sig_cands_ind)
                                        else:
                                                matched_num_matches[i] = 0
                        else:
                                matched_num_matches[i] = 0
        #Write output files
        endMCtime = time.time()-start_time-setup_time
        if verbose: print "Matching Done - Writing to files...Matching took " + str(endMCtime) + " seconds"
        FILE=open(outcat,'w')
        FILE2=open(outreg,'w')
        temp_ra_opts,temp_dec_opts,temp_ind_opts,temp_new_inds = np.zeros(3),np.zeros(3),np.zeros(3),np.array(['-00000000000001','-00000000000001','-00000000000001'])
        #Write ds9 region file preamble
        FILE2.write("# Region file format: DS9 version 4.1\n")
        FILE2.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\n')
        ci = 0
        for i in range(0,len(matched_num_matches)):
                #citemp is a parameter specific to the X-ray matching that creates an index based on 
                #only those sources detected with at least a 3 sigma significance
                if ((setup==Setup_Xray)&(usesig)):
                        citemp = 1
                        if sigmax[i] >= 3: 
                                citemp = ci
                                ci += 1
                temp_new_inds.fill(str(-1))
                temp_ind_opts[0] = matched_ind_opt1[i]
                temp_ind_opts[1] = matched_ind_opt2[i]
                temp_ind_opts[2] = matched_ind_opt3[i]
                for k in range(0,3):
                        if ((temp_ind_opts[k] != -1) and(temp_ind_opts[k] != "-1")):
                                temp_new_inds[k] = id_opt[int(temp_ind_opts[k])]
                #Write output data text file. Several of the parameters are setup-specific
                if setup==Setup_Xray:
                        FILE.write('%3i %9.5f %9.5f %8.5f %2i %9.5f %9.5f %15s %6.3f %7.1f %9.5f %9.5f %15s %6.3f %7.1f %9.5f %9.5f %15s %6.3f %7.1f %6.3f'%(matched_indX[i],matched_raX[i],matched_decX[i],matched_errX[i],matched_num_matches[i],matched_ra_opt1[i],matched_dec_opt1[i],temp_new_inds[0],matched_prob_opt1[i],matched_like_opt1[i],matched_ra_opt2[i],matched_dec_opt2[i],temp_new_inds[1],matched_prob_opt2[i],matched_like_opt2[i],matched_ra_opt3[i],matched_dec_opt3[i],temp_new_inds[2],matched_prob_opt3[i],matched_like_opt3[i],matched_prob_none[i]))
                        if usesig: FILE.write(' %3i %3.1f'%(citemp,sigmax[i]))
                        FILE.write('\n')
                elif setup==Setup_Radio:
                        FILE.write('%3i %9.5f %9.5f %8.5f %2i %9.5f %9.5f %15s %6.3f %7.1f %9.5f %9.5f %15s %6.3f %7.1f %9.5f %9.5f %15s %6.3f %7.1f %6.3f %7.3f\n'%(matched_indX[i],matched_raX[i],matched_decX[i],matched_errX[i],matched_num_matches[i],matched_ra_opt1[i],matched_dec_opt1[i],temp_new_inds[0],matched_prob_opt1[i],matched_like_opt1[i],matched_ra_opt2[i],matched_dec_opt2[i],temp_new_inds[1],matched_prob_opt2[i],matched_like_opt2[i],matched_ra_opt3[i],matched_dec_opt3[i],temp_new_inds[2],matched_prob_opt3[i],matched_like_opt3[i],matched_prob_none[i],radioflux[i]))
                else:
                        FILE.write('%3i %9.5f %9.5f %8.5f %2i %9.5f %9.5f %15s %6.3f %7.1f %9.5f %9.5f %15s %6.3f %7.1f %9.5f %9.5f %15s %6.3f %7.1f %6.3f\n'%(matched_indX[i],matched_raX[i],matched_decX[i],matched_errX[i],matched_num_matches[i],matched_ra_opt1[i],matched_dec_opt1[i],temp_new_inds[0],matched_prob_opt1[i],matched_like_opt1[i],matched_ra_opt2[i],matched_dec_opt2[i],temp_new_inds[1],matched_prob_opt2[i],matched_like_opt2[i],matched_ra_opt3[i],matched_dec_opt3[i],temp_new_inds[2],matched_prob_opt3[i],matched_like_opt3[i],matched_prob_none[i]))
                temp_ra_opts[0] = matched_ra_opt1[i]
                temp_ra_opts[1] = matched_ra_opt2[i]
                temp_ra_opts[2] = matched_ra_opt3[i]
                temp_dec_opts[0] = matched_dec_opt1[i]
                temp_dec_opts[1] = matched_dec_opt2[i]
                temp_dec_opts[2] = matched_dec_opt3[i]
                for j in range(0,int(matched_num_matches[i])):
                        #Write to ds9 region file
                        if j == 0: FILE2.write("circle(" + str(matched_raX[i]) + "," + str(matched_decX[i]) + "," + str(matched_errX[i]) + '")\n')
                        FILE2.write("circle(" + str(temp_ra_opts[j]) + "," + str(temp_dec_opts[j]) + "," + str(matched_errX[i]/5) + '") #color=red\n')
        FILE.close()
        FILE2.close()
        endtime = time.time()-start_time
        print "Total Elapsed time: " + str(endtime) + " seconds."
