from ROOT import TFile, TProfile2D,TCanvas, TGraph2D, TH2F, TH1F, TF1, gStyle, TPaveText, THStack

# gStyle.SetPalette(53)

def find_range(val, nbin):
    for i in range(nbin):
        if (val>(i*5)) and (val<((i+1)*5)):
            return i
    return None

def make_hist(tree, nbinx, xmin,xmax,nbiny,ymin,ymax):
    hist = TH2F('hist','hist',nbinx,xmin,xmax,nbiny,ymin,ymax)
    reshist = TH2F('reshist','reshist',nbinx,xmin,xmax,nbiny,ymin,ymax)
    myhist = TH2F('myhist','myhist',nbinx,xmin,xmax,nbiny,ymin,ymax)
    myreshist = TH2F('myreshist','myreshist',nbinx,xmin,xmax,nbiny,ymin,ymax)
    histlist = {}
    myhistlist = {}
    truehistlist = {}
    pophist = TH2F('pophist','pophist',nbinx,xmin,xmax,nbiny,ymin,ymax)
    relhist = TH2F('relhist','relhist',nbinx,xmin,xmax,nbiny,ymin,ymax)
    relreshist = TH2F('relreshist','relreshist',nbinx,xmin,xmax,nbiny,ymin,ymax)
    truehist = TH2F('truehist','truehist',nbinx,xmin,xmax,nbiny,ymin,ymax)
    truereshist = TH2F('truehist','truehist',nbinx,xmin,xmax,nbiny,ymin,ymax)
    print 'preparing hists'
    for i in range(nbinx):
        for j in range(nbiny):
            histlist['{}_{}'.format(i,j)] = TH1F('hist_{a}_{b}'.format(a=i*5,b=j*5),
                                                 'hist_ecal{a}_hcal{b}'.format(a=i*5,b=j*5),200,0,2)
            myhistlist['{}_{}'.format(i,j)] = TH1F('myhist_{a}_{b}'.format(a=i*5,b=j*5),
                                                 'myhist_ecal{a}_hcal{b}'.format(a=i*5,b=j*5),200,0,2)
            truehistlist['{}_{}'.format(i,j)] = TH1F('truehist_{a}_{b}'.format(a=i*5,b=j*5),
                                                     'truehist_ecal{a}_hcal{b}'.format(a=i*5,b=j*5),400,0,400)
    print 'reading tree'
    Nentry = tree.GetEntries()
    k = 0
    for event in tree:
        k+=1
        if k%10000==0:
            print 'event',k,'/',Nentry
        if (event.stdEcalib>407.3 and event.stdEcalib<407.4) or abs(event.eta)>1.5 or (event.ecal+event.hcal<150 or event.ecal+event.hcal>170):
            continue
        i = find_range(event.ecal,nbinx)
        j = find_range(event.hcal,nbiny)
        if (i is None) or (j is None) :
            continue
        histlist['{}_{}'.format(i,j)].Fill(event.stdEcalib/event.true)
        myhistlist['{}_{}'.format(i,j)].Fill(event.myEcalib/event.true)
        truehistlist['{}_{}'.format(i,j)].Fill(event.true)

    print 'start fitting'
    gauss = TF1('gauss','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
    for i in range(nbinx):
        for j in range(nbiny):
            if j+i>60:
                relhist.SetBinContent(i+1,j+1,-99)
                relreshist.SetBinContent(i+1,j+1,-99)
                continue
            print 'fitting xbin:',i,'ybin :',j
            if histlist['{}_{}'.format(i,j)].Integral()>0.:
                gauss.SetParameters(myhistlist['{}_{}'.format(i,j)].GetMaximum(),
                                    myhistlist['{}_{}'.format(i,j)].GetMean(),
                                    myhistlist['{}_{}'.format(i,j)].GetRMS())
                histlist['{}_{}'.format(i,j)].Fit(gauss)
                hist.SetBinContent(i+1,j+1,gauss.GetParameter(1))
                hist.SetBinError(i+1,j+1,gauss.GetParError(1))
                reshist.SetBinContent(i+1,j+1,abs(gauss.GetParameter(2)))
                reshist.SetBinError(i+1,j+1,gauss.GetParError(2))
                pophist.SetBinContent(i+1,j+1,myhistlist['{}_{}'.format(i,j)].Integral()/Nentry)
                gauss.SetParameters(truehistlist['{}_{}'.format(i,j)].GetMaximum(),
                                    truehistlist['{}_{}'.format(i,j)].GetMean(),
                                    truehistlist['{}_{}'.format(i,j)].GetRMS())
                truehistlist['{}_{}'.format(i,j)].Fit(gauss)
                truehist.SetBinContent(i+1,j+1,gauss.GetParameter(1))
                truereshist.SetBinContent(i+1,j+1,abs(gauss.GetParameter(2)))
            else:
                hist.SetBinContent(i+1,j+1,0)
                reshist.SetBinContent(i+1,j+1,0)
            if myhistlist['{}_{}'.format(i,j)].Integral()>0.:
                gauss.SetParameters(myhistlist['{}_{}'.format(i,j)].GetMaximum(),
                                    myhistlist['{}_{}'.format(i,j)].GetMean(),
                                    myhistlist['{}_{}'.format(i,j)].GetRMS())
                myhistlist['{}_{}'.format(i,j)].Fit(gauss)
                myhist.SetBinContent(i+1,j+1,gauss.GetParameter(1))
                myhist.SetBinError(i+1,j+1,gauss.GetParError(1))
                myreshist.SetBinContent(i+1,j+1,abs(gauss.GetParameter(2)))
                myreshist.SetBinError(i+1,j+1,gauss.GetParError(2))
            else:
                myhist.SetBinContent(i+1,j+1,0)
                myreshist.SetBinContent(i+1,j+1,0)
            if histlist['{}_{}'.format(i,j)].Integral()>0. and myhistlist['{}_{}'.format(i,j)].Integral()>0.:
                relhist.SetBinContent(i+1,j+1,hist.GetBinContent(i+1,j+1)-myhist.GetBinContent(i+1,j+1))
                relreshist.SetBinContent(i+1,j+1,reshist.GetBinContent(i+1,j+1)-myreshist.GetBinContent(i+1,j+1))
    hist.GetZaxis().SetRangeUser(0.8,1.2)
    myhist.GetZaxis().SetRangeUser(0.8,1.2)
    relhist.GetZaxis().SetRangeUser(-0.2,0.2)
    truehist.GetZaxis().SetRangeUser(0,300)
    truereshist.GetZaxis().SetRangeUser(0,50)
    hist.SetStats(0)
    truehist.SetStats(0)
    truereshist.SetStats(0)
    myhist.SetStats(0)
    pophist.SetStats(0)
    relhist.SetStats(0)
    relreshist.SetStats(0)
    hist.GetXaxis().SetTitle('E_{ecal}')
    hist.GetYaxis().SetTitle('E_{hcal}')
    truehist.GetXaxis().SetTitle('E_{ecal}')
    truehist.GetYaxis().SetTitle('E_{hcal}')
    truereshist.GetXaxis().SetTitle('E_{ecal}')
    truereshist.GetYaxis().SetTitle('E_{hcal}')
    relhist.GetXaxis().SetTitle('E_{ecal}')
    relhist.GetYaxis().SetTitle('E_{hcal}')
    relreshist.GetXaxis().SetTitle('E_{ecal}')
    relreshist.GetYaxis().SetTitle('E_{hcal}')
    pophist.GetXaxis().SetTitle('E_{ecal}')
    pophist.GetYaxis().SetTitle('E_{hcal}')
    myhist.GetXaxis().SetTitle('E_{ecal}')
    myhist.GetYaxis().SetTitle('E_{hcal}')
    hist.SetTitle('gaussian mean of (standard Ecalib/Etrue)')
    relhist.SetTitle('gaussian mean difference of (Ecalib/Etrue) (std-KNN)')
    relreshist.SetTitle('gaussian standard deviation difference of (Ecalib/Etrue) (std-KNN)')
    pophist.SetTitle('% of total population')
    myhist.SetTitle('gaussian mean of (KNN Ecalib/Etrue)')
    truehist.SetTitle('mean of Etrue')
    truereshist.SetTitle('deviation of Etrue')
    
    reshist.GetZaxis().SetRangeUser(0.,0.5)
    myreshist.GetZaxis().SetRangeUser(0.,0.5)
    relreshist.GetZaxis().SetRangeUser(-0.05,0.05)
    reshist.SetStats(0)
    myreshist.SetStats(0)
    reshist.GetXaxis().SetTitle('E_{ecal}')
    reshist.GetYaxis().SetTitle('E_{hcal}')
    myreshist.GetXaxis().SetTitle('E_{ecal}')
    myreshist.GetYaxis().SetTitle('E_{hcal}')
    reshist.SetTitle('gaussian standard deviation of (standard Ecalib/Etrue)')
    myreshist.SetTitle('gaussian standard deviation of (KNN Ecalib/Etrue)')

    return hist, myhist, histlist, myhistlist, reshist, myreshist, pophist, relhist, relreshist, truehist, truehistlist, truereshist


special_case = '_ecalhcal_150_170_2'

doeta = True #(if or if not eta : the value to exclude changes in line 40!!!)

# fil = TFile('all_calibs'+('_eta' if doeta else '')+'.root')
fil = TFile('all_calibs_01_02_direct.root')
tree = fil.Get('s')
hist, myhist, histlist, myhistlist, reshist, myreshist, pophist, relhist, relreshist, truehist, truehistlist,truereshist = make_hist(tree,60,0,300,60,0,300)

can1 = TCanvas()
hist.Draw('colz')
can1.SaveAs('plots/gauss_mean_std'+('_eta' if doeta else '')+(special_case)+'.png')
can2 = TCanvas()
myhist.Draw('colz')
can2.SaveAs('plots/gauss_mean_KNN'+('_eta' if doeta else '')+(special_case)+'.png')

can12 = TCanvas()
hist.GetZaxis().SetRangeUser(0.95,1.05)
hist.Draw('colz')
can12.SaveAs('plots/gauss_mean_std_zzoom'+('_eta' if doeta else '')+(special_case)+'.png')
can11 = TCanvas()
myhist.GetZaxis().SetRangeUser(0.95,1.05)
myhist.Draw('colz')
can11.SaveAs('plots/gauss_mean_KNN_zzoom'+('_eta' if doeta else '')+(special_case)+'.png')

can3 = TCanvas()
reshist.Draw('colz')
can3.SaveAs('plots/gauss_dev_std'+('_eta' if doeta else '')+(special_case)+'.png')
can4 = TCanvas()
myreshist.Draw('colz')
can4.SaveAs('plots/gauss_dev_KNN'+('_eta' if doeta else '')+(special_case)+'.png')

can5 = TCanvas()
relreshist.Draw('colz')
can5.SaveAs('plots/gauss_dev_rel'+('_eta' if doeta else '')+(special_case)+'.png')
can6 = TCanvas()
relhist.Draw('colz')
can6.SaveAs('plots/gauss_mean_rel'+('_eta' if doeta else '')+(special_case)+'.png')
can7 = TCanvas()
pophist.Draw('lego')
can7.SaveAs('plots/population'+('_eta' if doeta else '')+(special_case)+'.png')
can8 = TCanvas()
pophist.Draw('lego')
can8.SetLogz()
can8.SaveAs('plots/population_log'+('_eta' if doeta else '')+(special_case)+'.png')

can9 = TCanvas()
truehist.Draw('colz')
can9.SaveAs('plots/true'+('_eta' if doeta else '')+(special_case)+'.png')
can10 = TCanvas()
truereshist.Draw('colz')
can10.SaveAs('plots/true_res'+('_eta' if doeta else '')+(special_case)+'.png')


if special_case == '_ecalhcal_150_170':
    stdhist = TH1F('testhist','sum of hists for 150<ecal+hcal<170 (std)',200,0,2)
    stdhist.GetXaxis().SetTitle('E_{calib}/E_{true}')
    stdhist.GetYaxis().SetTitle('N_{events}')
    knnhist = TH1F('testhist','sum of hists for 150<ecal+hcal<170 (KNN)',200,0,2)
    knnhist.GetXaxis().SetTitle('E_{calib}/E_{true}')
    knnhist.GetYaxis().SetTitle('N_{events}')
    lowstdhist = TH1F('testhist','sum of hists for 150<ecal+hcal<170 (std low ecal)',200,0,2)
    lowstdhist.GetXaxis().SetTitle('E_{calib}/E_{true}')
    lowstdhist.GetYaxis().SetTitle('N_{events}')
    lowknnhist = TH1F('testhist','sum of hists for 150<ecal+hcal<170 (KNN low ecal)',200,0,2)
    lowknnhist.GetXaxis().SetTitle('E_{calib}/E_{true}')
    lowknnhist.GetYaxis().SetTitle('N_{events}')
    upstdhist = TH1F('testhist','sum of hists for 150<ecal+hcal<170 (std high ecal)',200,0,2)
    upstdhist.GetXaxis().SetTitle('E_{calib}/E_{true}')
    upstdhist.GetYaxis().SetTitle('N_{events}')
    upknnhist = TH1F('testhist','sum of hists for 150<ecal+hcal<170 (KNN high ecal)',200,0,2)
    upknnhist.GetXaxis().SetTitle('E_{calib}/E_{true}')
    upknnhist.GetYaxis().SetTitle('N_{events}')
    for i in range(60):
        for j in range(60):
            if i+j>29 and i+j<35:
                stdhist.Add(histlist['{}_{}'.format(i,j)])
                knnhist.Add(myhistlist['{}_{}'.format(i,j)])
                if i>(j+14):
                    upstdhist.Add(histlist['{}_{}'.format(i,j)])
                    upknnhist.Add(myhistlist['{}_{}'.format(i,j)])
                elif i<(j-23) and i>1:
                    lowstdhist.Add(histlist['{}_{}'.format(i,j)])
                    lowknnhist.Add(myhistlist['{}_{}'.format(i,j)])
    gauss1 = TF1('gauss1','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
    gauss1.SetParameters(stdhist.GetMaximum(),stdhist.GetMean(),stdhist.GetRMS())
    stdhist.Fit('gauss1')
    stdhist.SetTitle('mean '+str(gauss1.GetParameter(1))+' '+'dev '+str(gauss1.GetParameter(2)))
    stdhist.SetFillColor(3)

    can20 = TCanvas()
    stdhist.Draw()
    #pav1 = TPaveText(.1,.7,.5,.9)
    #pav1.AddText('mean '+str(gauss1.GetParameter(1)))
    #pav1.AddText('dev '+str(gauss1.GetParameter(2)))
    #pav1.Draw()
    can20.SaveAs('plots/resp_std_tot'+('_eta' if doeta else '')+(special_case)+'.png')
    
    gauss2 = TF1('gauss2','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
    gauss2.SetParameters(knnhist.GetMaximum(),knnhist.GetMean(),knnhist.GetRMS())
    knnhist.Fit('gauss2')
    can21 = TCanvas()
    knnhist.SetTitle('mean '+str(gauss2.GetParameter(1))+' '+'dev '+str(gauss2.GetParameter(2)))
    knnhist.Draw()
    #pav2 = TPaveText(.1,.7,.5,.9)
    #pav2.AddText('mean '+str(gauss2.GetParameter(1)))
    #pav2.AddText('dev '+str(gauss2.GetParameter(2)))
    #pav2.Draw()
    can21.SaveAs('plots/resp_knn_tot'+('_eta' if doeta else '')+(special_case)+'.png')

    lowgauss1 = TF1('lowgauss1','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
    lowgauss1.SetParameters(lowstdhist.GetMaximum(),lowstdhist.GetMean(),lowstdhist.GetRMS())
    lowstdhist.Fit('lowgauss1')
    can22 = TCanvas()
    lowstdhist.SetTitle('mean '+str(lowgauss1.GetParameter(1))+' '+'dev '+str(lowgauss1.GetParameter(2)))
    lowstdhist.SetFillColor(4)
    lowstdhist.Draw()
    #pav3 = TPaveText(.1,.7,.5,.9)
    #pav3.AddText('mean '+str(lowgauss1.GetParameter(1)))
    #pav3.AddText('dev '+str(lowgauss1.GetParameter(2)))
    #pav3.Draw()
    can22.SaveAs('plots/resp_std_low'+('_eta' if doeta else '')+(special_case)+'.png')
    lowgauss2 = TF1('lowgauss2','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
    lowgauss2.SetParameters(lowknnhist.GetMaximum(),lowknnhist.GetMean(),lowknnhist.GetRMS())
    lowknnhist.Fit('lowgauss2')
    can23 = TCanvas()
    lowknnhist.SetTitle('mean '+str(lowgauss2.GetParameter(1))+' '+'dev '+str(lowgauss2.GetParameter(2)))
    lowknnhist.Draw()
    #pav4 = TPaveText(.1,.7,.5,.9)
    #pav4.AddText('mean '+str(lowgauss2.GetParameter(1)))
    #pav4.AddText('dev '+str(lowgauss2.GetParameter(2)))
    #pav4.Draw()
    can23.SaveAs('plots/resp_knn_low'+('_eta' if doeta else '')+(special_case)+'.png')

    upgauss1 = TF1('upgauss1','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
    upgauss1.SetParameters(upstdhist.GetMaximum(),upstdhist.GetMean(),upstdhist.GetRMS())
    upstdhist.Fit('upgauss1')
    can24 = TCanvas()
    upstdhist.SetTitle('mean '+str(upgauss1.GetParameter(1))+' '+'dev '+str(upgauss1.GetParameter(2)))
    upstdhist.SetFillColor(5)
    upstdhist.Draw()
    #pav5 = TPaveText(.1,.7,.5,.9)
    #pav5.AddText('mean '+str(upgauss1.GetParameter(1)))
    #pav5.AddText('dev '+str(upgauss1.GetParameter(2)))
    #pav5.Draw()
    can24.SaveAs('plots/resp_std_up'+('_eta' if doeta else '')+(special_case)+'.png')
    upgauss2 = TF1('upgauss2','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
    upgauss2.SetParameters(upknnhist.GetMaximum(),upknnhist.GetMean(),upknnhist.GetRMS())
    upknnhist.Fit('upgauss2')
    can25 = TCanvas()
    upknnhist.SetTitle('mean '+str(upgauss2.GetParameter(1))+' '+'dev '+str(upgauss2.GetParameter(2)))
    upknnhist.Draw()
    #pav6 = TPaveText(.1,.7,.5,.9)
    #pav6.AddText('mean '+str(upgauss2.GetParameter(1)))
    #pav6.AddText('dev '+str(upgauss2.GetParameter(2)))
    #pav6.Draw()
    can25.SaveAs('plots/resp_knn_up'+('_eta' if doeta else '')+(special_case)+'.png')

    stdhist.Add(upstdhist,-1)
    stdhist.Add(lowstdhist,-1)
    stdstack = THStack('stdstack', 'mean '+str(gauss1.GetParameter(1))+' '+'dev '+str(gauss1.GetParameter(2)))
    stdstack.Add(upstdhist)
    stdstack.Add(lowstdhist)
    stdstack.Add(stdhist)
    can30 = TCanvas()
    stdstack.Draw()
    gauss1.Draw('same')
    can30.SaveAs('plots/stack_std'+('_eta' if doeta else '')+(special_case)+'.png')

def savecan(h, name):
    tmpcan = TCanvas()
    h.Draw()
    tmpcan.SaveAs('plots/stack_2000neighbours_direct/'+name+'.png')

def draw_stack(hlist, listname):
    icolor = 1
    stck = THStack('stck', '')
    gauss = TF1('gauss','[0]*exp(-0.5*((x-[1])/[2])**2)',0,2)
    _hists = []
    _hists.append(TH1F('testhist_{}'.format(0),'bla',200,0,2))
    _hists[-1].GetXaxis().SetTitle('E_{calib}/E_{true}')
    _hists[-1].GetYaxis().SetTitle('N_{events}')
    _hists[-1].SetFillColor(icolor)
    icolor+=1
    for i in [29,30,31,32,33]:
        _hists[-1].Add(hlist['0_{}'.format(i)])
    gauss.SetParameters(_hists[-1].GetMaximum(),_hists[-1].GetMean(),_hists[-1].GetRMS())
    _hists[-1].Fit('gauss')
    _hists[-1].SetTitle('Eecal<5 '+'mean '+str(gauss.GetParameter(1))+' '+'dev '+str(gauss.GetParameter(2)))
    savecan(_hists[-1],listname+'0')
    # stck.Add(_hists[-1])
    totpop = sum([x.Integral() for n,x in hlist.iteritems() if n[0]!='0'])
    npop = totpop/10.
    x = 1
    y = 33 - (x+5)
    for i in range(10):
        start = x*5
        _hists.append(TH1F('testhist_{}'.format(i),'bla',200,0,2))
        _hists[-1].GetXaxis().SetTitle('E_{calib}/E_{true}')
        _hists[-1].GetYaxis().SetTitle('N_{events}')
        _hists[-1].SetFillColor(icolor)
        icolor+=1
        while _hists[-1].Integral()<npop:
            # import pdb;pdb.set_trace()
            if y+x>33:
                x+=1
                y=33 - (x+5)
                if y<0:
                    y= 0
            if x>33:
                break
            _hists[-1].Add(hlist['{}_{}'.format(x,y)])
            y+=1
        finish = x*5
        gauss.SetParameters(_hists[-1].GetMaximum(),_hists[-1].GetMean(),_hists[-1].GetRMS())
        _hists[-1].Fit('gauss')
        _hists[-1].SetTitle('{}<=Eecal<={} '.format(start,finish)+'mean '+str(gauss.GetParameter(1))+' '+'dev '+str(gauss.GetParameter(2)))
        savecan(_hists[-1],listname+str(i))
        # stck.Add(_hists[-1])
    ###
    stck.Add(_hists[1])
    stck.Add(_hists[10])
    stck.Add(_hists[2])
    stck.Add(_hists[9])
    stck.Add(_hists[3])
    stck.Add(_hists[8])
    stck.Add(_hists[4])
    stck.Add(_hists[7])
    stck.Add(_hists[5])
    stck.Add(_hists[6])
    ###
    hlast = stck.GetStack().Last()
    gauss.SetParameters(hlast.GetMaximum(),hlast.GetMean(),hlast.GetRMS())
    hlast.Fit('gauss')
    stck.SetTitle('mean '+str(gauss.GetParameter(1))+' '+'dev '+str(gauss.GetParameter(2)))
    savecan(stck,listname+'stack')

draw_stack(histlist, 'std')

draw_stack(myhistlist, 'knn')
    
