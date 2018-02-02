from ROOT import TFile, TProfile2D,TCanvas, TGraph2D, TH2F, TH1F, TF1, TLegend

def find_range(val):
    for i in range(60):
        if (val>(i)) and (val<((i+1))):
            return i
    return None

def make_hist(tree):
    stdhist = TH1F('stdhist','',60,0,300)
    myhist = TH1F('myhist','',60,0,300)
    stdreshist = TH1F('stdreshist','',60,0,300)
    myreshist = TH1F('myreshist','',60,0,300)
    histlist = {}
    myhistlist = {}
    for i in range(60):
        histlist['{}_'.format(i)] = TH1F('hist_{a}'.format(a=i),
                                         'hist_{a}'.format(a=i),100,0,2)
        myhistlist['{}_'.format(i)] = TH1F('myhist_{a}'.format(a=i),
                                           'myhist_{a}'.format(a=i),100,0,2)
    print 'reading tree'
    Nentry = tree.GetEntries()
    k = 0
    for event in tree:
        k+=1
        if k%10000==0:
            print 'event',k,'/',Nentry
        if (event.stdEcalib>407.3 and event.stdEcalib<407.4) or abs(event.eta)>1.5 or event.ecal+event.hcal>300 and event.ecal>20 and event.hcal>20:
            continue
        i = find_range(event.true)
        if i>=60:
            continue
        if (i is None):
            continue
        histlist['{}_'.format(i)].Fill(event.stdEcalib/event.true)
        myhistlist['{}_'.format(i)].Fill(event.myEcalib/event.true)

    print 'start fitting'
    gauss = TF1('gauss','[0]*exp(-0.5*((x-[1])/[2])**2)',0,10)
    for i in range(60):
        print 'fitting',i,'/60'
        if histlist['{}_'.format(i)].Integral()>0.:
            gauss.SetParameters(myhistlist['{}_'.format(i)].GetMaximum(),
                                myhistlist['{}_'.format(i)].GetMean(),
                                myhistlist['{}_'.format(i)].GetRMS())
            histlist['{}_'.format(i)].Fit(gauss)
            stdhist.SetBinContent(i+1,gauss.GetParameter(1))
            stdhist.SetBinError(i+1,gauss.GetParError(1))
            stdreshist.SetBinContent(i+1,abs(gauss.GetParameter(2)))
            stdreshist.SetBinError(i+1,gauss.GetParError(2))
        else:
            stdhist.SetBinContent(i+1,0)
            stdreshist.SetBinContent(i+1,0)
        if myhistlist['{}_'.format(i)].Integral()>0.:
            gauss.SetParameters(myhistlist['{}_'.format(i)].GetMaximum(),
                                myhistlist['{}_'.format(i)].GetMean(),
                                myhistlist['{}_'.format(i)].GetRMS())
            myhistlist['{}_'.format(i)].Fit(gauss)
            myhist.SetBinContent(i+1,gauss.GetParameter(1))
            myhist.SetBinError(i+1,gauss.GetParError(1))
            myreshist.SetBinContent(i+1,abs(gauss.GetParameter(2)))
            myreshist.SetBinError(i+1,gauss.GetParError(2))
        else:
            myhist.SetBinContent(i+1,0)
            myreshist.SetBinContent(i+1,0)
    stdhist.SetStats(0)
    myhist.SetStats(0)
    stdhist.GetXaxis().SetTitle('E_{true}')
    stdhist.GetYaxis().SetTitle('response')
    stdhist.GetYaxis().SetRangeUser(0,2)
    myhist.GetXaxis().SetTitle('E_{true}')
    myhist.GetYaxis().SetTitle('KNN resp')
    myhist.GetYaxis().SetRangeUser(0,2)
    stdreshist.SetStats(0)
    myreshist.SetStats(0)
    stdreshist.GetXaxis().SetTitle('E_{true}')
    stdreshist.GetYaxis().SetTitle('resolution')
    stdreshist.GetYaxis().SetRangeUser(0,2)
    myreshist.GetXaxis().SetTitle('E_{true}')
    myreshist.GetYaxis().SetTitle('KNN res')
    myreshist.GetYaxis().SetRangeUser(0,2)

    return stdhist, myhist, histlist, myhistlist , stdreshist, myreshist

    
doeta = False # if false use (event.stdEcalib>434.55 and event.stdEcalib<434.56)
do_01_02 = True

fil = TFile('all_calibs'+('_eta' if doeta else '')+('_01_02_direct' if do_01_02 else '')+'.root')
tree = fil.Get('s')
hist, myhist, histlist, myhistlist , reshist, myreshist= make_hist(tree)

can1 = TCanvas()
hist.Draw()
myhist.SetLineColor(2)
myhist.Draw("same")
leg1 = TLegend(0.7,0.7,0.9,0.9)
leg1.AddEntry(hist,"Standard method","l")
leg1.AddEntry(myhist,"KNN method","l")
leg1.Draw()
can1.SaveAs('gauss_mean_vstrue'+('_eta' if doeta else '')+('_01_02_direct' if do_01_02 else '')+'.png')
can2 = TCanvas()
reshist.Draw()
myreshist.SetLineColor(2)
myreshist.Draw("same")
leg2 = TLegend(0.7,0.7,0.9,0.9)
leg2.AddEntry(hist,"Standard method","l")
leg2.AddEntry(myhist,"KNN method","l")
leg2.Draw()
can2.SaveAs('gauss_res_vstrue'+('_eta' if doeta else '')+('_01_02_direct' if do_01_02 else '')+'.png')
