/*************************************************************************/
/*********************** Solve evolution of rho (h) **********************/
/*************************************************************************/
/*********************** STEPHEN LAUGHTON 13/5/95 ************************/
/*************************** cc code.c -lm -O ****************************/
/*************************************************************************/
/******** integrals performed using Bode's and 6 pt Simpson rules ********/
/*************************************************************************/


#define PI 3.141592654
#define TWOoPI 0.636619772
#define TPI 6.283185307
#define INTERVAL (int)(3*5)                         /****** odd x*5 ******/
#define INTSQUARE (INTERVAL*INTERVAL)               /* INTERVAL*INTERVAL */
#define PARTITION (int)(6*4)                        /**** even x*4 *******/
#define HMAX 4.0
#define HMIN -4.0
#define HMAX_MIN HMAX-(HMIN)                        /*** HMAX-HMIN *******/
#define HM_MoP (HMAX_MIN)/PARTITION
#define K (double)0.0
#define OK (double)(1.0-K*K)
#define KO (double)((1.0-K*K)/(1.0+K*K))
#define NR_END 1
#define ACCURACY (double)0.000001
#define ISTEP (int)6                                /******* even ********/
#define GRAD (double)0.04
#define RTEST 0.003
#define NSTEPS (int)100



#include <stdio.h>
#include <math.h>
#include "extra/extmath.h"
#include "extra/matrices.h"


void calcstuff(),calccoefficients_wy(),calcmmur(),calcrho(),check(),calct();

double saddle(),calcq();


double J_0,J_1,Js,T,beta;
double chiplus[PARTITION+1],chiminus[PARTITION+1];
double rhoplus[PARTITION+1],rhominus[PARTITION+1];
double calcrhoplus[PARTITION+1],calcrhominus[PARTITION+1];
double rhoplussr[PARTITION+1],rhominussr[PARTITION+1];
double rhoplush[PARTITION+1],rhominush[PARTITION+1];
double tanhplus,tanhminus,tanhsigmarho,htanh,h_tanh;
double weights[]={0.311111111,1.422222222,0.5333333333,1.422222222,0.311111111};
double sweight[]={0.329861111,1.302083333,0.868055555,0.868055555,1.302083333,0.329861111};


double ***measureplus,***measureminus;
double ***f3tensor();

double m,mu,r,Theta_0,Theta_1,E,rterm,tterm,newr1,newt1,newmu;


main()
{
  FILE *fp;
  int iteration,k,l,n;
  float in;
  double time,deltat,hJm,hmJm,hpJm,thbh,ex,test;
  double Phi_1,Phi_2plus,Phi_2minus,Phi_3plus[PARTITION+1];
  double Phi_3minus[PARTITION+1],Phi_4,interm[2*INTERVAL+2];
  double q,lambda0s,lambda1s;
  double deriv1plus,deriv1minus,deriv2plus,deriv2minus;
  double deltarhoplus[PARTITION+1],deltarhominus[PARTITION+1];

  char flowfile[12],distfile[12],chifile[12];

  measureplus=f3tensor(0,2*INTERVAL+1,0,2*INTERVAL+1,0,PARTITION);
  measureminus=f3tensor(0,2*INTERVAL+1,0,2*INTERVAL+1,0,PARTITION);



/**************************** input parameters ***************************/

  printf("Enter value of J_0 \n");
  scanf("%f",&in);
  J_0=(double)in;
  printf("J = 1 \n");
  J_1=1.0;           Js=J_1*J_1;
  printf("Enter value of T \n");
  scanf("%f",&in);
  T=(double)in;
  if(T!=0.0)
    beta=1.0/T;
  printf("Enter initial m \n");
  scanf("%f",&in);
  m=(double)in;
  
  sprintf(flowfile,"flow.%d.%d.%d",(int)(T),(int)(J_0),(int)(m*10.0));
  fp=fopen(flowfile,"w");
  fprintf(fp,"INTERVAL %d PARTITION %d\n",INTERVAL,PARTITION);
  fprintf(fp,"ACCURACY %f GRAD %f RTEST %f\n",ACCURACY,GRAD,RTEST);
  fprintf(fp,"K = %f\n",(float)K);
  fprintf(fp,"J_0 = %f\n",(float)J_0);
  fprintf(fp,"T= %f\n",(float)T);
  fprintf(fp,"m initial = %f\n",(float)m);
  fprintf(fp,"t  m  E  dm/dt  dE/dt q AT\n");
  fclose(fp);

  sprintf(distfile,"dist.%d.%d.%d",(int)(T),(int)(J_0),(int)(m*10.0));
  fp=fopen(distfile,"w");
  fprintf(fp,"INTERVAL %d PARTITION %d\n",INTERVAL,PARTITION);
  fprintf(fp,"ACCURACY %f GRAD %f RTEST %f\n",ACCURACY,GRAD,RTEST);
  fprintf(fp,"K = %f\n",(float)K);
  fprintf(fp,"J_0 = %f\n",(float)J_0);
  fprintf(fp,"T= %f\n",(float)T);
  fprintf(fp,"m initial = %f\n",(float)m);
  fclose(fp);

  sprintf(chifile,"chi.%d.%d.%d",(int)(T),(int)(J_0),(int)(m*10.0));

/*************************************************************************/



/************************ find initial chi *******************************/

  for(k=0;k<=PARTITION;k++) {
    chiplus[k]=0.5*(1.0+m);    chiminus[k]=0.5*(1.0-m);
    hJm=(HMIN+k*HM_MoP)-J_0*m;
    ex=exp(-hJm*hJm/(2.0*Js));
    rhoplus[k]=chiplus[k]*ex/sqrt(TPI*Js);
    rhominus[k]=chiminus[k]*ex/sqrt(TPI*Js);
  }

  test=0.0;
  for(l=0;l<(PARTITION/4);l++) {
    for(n=0;n<5;n++) {
      k=l*4+n;
      test+=weights[n]*(rhoplus[k]+rhominus[k]);
    }
  }  
  test*=HM_MoP;

  for(k=0;k<=PARTITION;k++) {
    rhoplus[k]/=test;
    rhominus[k]/=test;
  }
  q=m*m;   mu=0.0;   r=0.0;   Theta_0=0.0;   Theta_1=0.0;
  rterm=q; tterm=0.0;
  calcstuff(q);

/*************************************************************************/


  time=0.0;
  do {
    if(time!=0.0)  q=saddle(q);

    if((time-(int)time)<=0.001) {
      fp=fopen(distfile,"a");
      fprintf(fp,"t %f\n",time);
      fprintf(fp,"h rhoplus rhominus \n");
      for(k=0;k<=PARTITION;k++) {
	fprintf(fp,"%f %f %f\n",HMIN+k*HM_MoP,rhoplus[k],rhominus[k]);
      }
      fclose(fp);

      fp=fopen(chifile,"w");
      fprintf(fp,"t %f\n",time);
      fprintf(fp,"h chiplus chiminus \n");
      for(k=0;k<=PARTITION;k++) {
	fprintf(fp,"%f %f %f\n",HMIN+k*HM_MoP,chiplus[k],chiminus[k]);
      }
      fprintf(fp,"tterm %f rterm %f q %f\n",tterm,rterm,q);
      fclose(fp); 
    }


/************** evaluate evolution equation coefficients *****************/

    calccoefficients_wy(q,time,flowfile);

    lambda0s=(1.0-2*q)/(Js*(1.0-q)*(1.0-q));
    lambda1s=q/(Js*(1.0-q)*(1.0-q));


    Phi_1=1.0-Js*lambda0s*(tanhplus-tanhminus)-Js*lambda1s*tanhsigmarho;
    Phi_2plus=J_0*m-J_0*(tanhplus+tanhminus)-Js*KO*(lambda0s*htanh+lambda1s*h_tanh)
      +Js*2.0*KO*((Theta_0+Theta_1)*(tanhplus-tanhminus)-Theta_1*tanhsigmarho);
    Phi_2minus=J_0*m-J_0*(tanhplus+tanhminus)+Js*KO*(lambda0s*htanh+lambda1s*h_tanh)
      -Js*2.0*KO*((Theta_0+Theta_1)*(tanhplus-tanhminus)-Theta_1*tanhsigmarho);
    for(k=0;k<=PARTITION;k++) {
      Phi_3plus[k]=Js*KO*rhoplussr[k]*((lambda0s+2*lambda1s)*h_tanh-lambda1s*htanh)
	+Js*rhoplush[k]*((lambda0s+2*lambda1s)*tanhsigmarho-lambda1s*(tanhplus-tanhminus))
	  -Js*2.0*KO*rhoplussr[k]*((Theta_0-Theta_1)*tanhsigmarho+Theta_1*(tanhplus-tanhminus));
      Phi_3minus[k]=Js*KO*rhominussr[k]*((lambda0s+2*lambda1s)*h_tanh-lambda1s*htanh)
	+Js*rhominush[k]*((lambda0s+2*lambda1s)*tanhsigmarho-lambda1s*(tanhplus-tanhminus))
	  -Js*2.0*KO*rhominussr[k]*((Theta_0-Theta_1)*tanhsigmarho+Theta_1*(tanhplus-tanhminus));
    }
    Phi_4=Js*(1.0-tanhplus+tanhminus);


/***************************** advance solution **************************/

    deltat=HM_MoP*HM_MoP/(10.0*Phi_4);
    if(((int)time)!=((int)(time+deltat))) deltat=1.0*(int)(time+deltat)-time;
    time+=deltat;


    for(k=0;k<=PARTITION;k++) {
      hJm=HMIN+k*HM_MoP-J_0*m;
      hmJm=hJm-HM_MoP;
      hpJm=hJm+HM_MoP;

/***************************** find terms ********************************/

      if(k==0) {
	deriv1plus=(Phi_1*(hpJm*rhoplus[k+1]-hJm*rhoplus[k])
		    +Phi_2plus*(rhoplus[k+1]-rhoplus[k])
		    +(Phi_3plus[k+1]-Phi_3plus[k]))/(HM_MoP);
	deriv1minus=(Phi_1*(hpJm*rhominus[k+1]-hJm*rhominus[k])
		     +Phi_2minus*(rhominus[k+1]-rhominus[k])
		     +(Phi_3minus[k+1]-Phi_3minus[k]))/(HM_MoP);

	deriv2plus=Phi_4*(rhoplus[k+1]-2.0*rhoplus[k])/(HM_MoP*HM_MoP);
	deriv2minus=Phi_4*(rhominus[k+1]-2.0*rhominus[k])/(HM_MoP*HM_MoP);
      }

      if(k>0 && k <PARTITION) {
	deriv1plus=0.5*(Phi_1*(hpJm*rhoplus[k+1]-hmJm*rhoplus[k-1])
			+Phi_2plus*(rhoplus[k+1]-rhoplus[k-1])
			+(Phi_3plus[k+1]-Phi_3plus[k-1]))/(HM_MoP);
	deriv1minus=0.5*(Phi_1*(hpJm*rhominus[k+1]-hmJm*rhominus[k-1])
			 +Phi_2minus*(rhominus[k+1]-rhominus[k-1])
			 +(Phi_3minus[k+1]-Phi_3minus[k-1]))/(HM_MoP);

	deriv2plus=Phi_4*(rhoplus[k+1]-2.0*rhoplus[k]+rhoplus[k-1])/(HM_MoP*HM_MoP);
	deriv2minus=Phi_4*(rhominus[k+1]-2.0*rhominus[k]+rhominus[k-1])/(HM_MoP*HM_MoP);
      }

      if(k==PARTITION) {
	deriv1plus=(Phi_1*(hJm*rhoplus[k]-hmJm*rhoplus[k-1])
		    +Phi_2plus*(rhoplus[k]-rhoplus[k-1])
		    +(Phi_3plus[k]-Phi_3plus[k-1]))/(HM_MoP);
	deriv1minus=(Phi_1*(hJm*rhominus[k]-hmJm*rhominus[k-1])
		     +Phi_2minus*rhominus[k]-(rhominus[k-1])
		     +(Phi_3minus[k]-Phi_3minus[k-1]))/(HM_MoP);

	deriv2plus=Phi_4*(-2.0*rhoplus[k]+rhoplus[k-1])/(HM_MoP*HM_MoP);
	deriv2minus=Phi_4*(-2.0*rhominus[k]+rhominus[k-1])/(HM_MoP*HM_MoP);
      }

/*************************************************************************/

      if(T!=0.0)
	thbh=tanh(beta*(HMIN+k*HM_MoP));
      else {
	if(HMIN+k*HM_MoP>0.0) thbh=1.0;
	if(HMIN+k*HM_MoP<=0.0) thbh=-1.0;
	if(HMIN+k*HM_MoP==0.0) thbh=0.0;
      }
      deltarhoplus[k]=deltat*(0.5*((1.0+thbh)*rhominus[k]-(1.0-thbh)*rhoplus[k])
			      +deriv1plus+deriv2plus);
      deltarhominus[k]=deltat*(0.5*((1.0-thbh)*rhoplus[k]-(1.0+thbh)*rhominus[k])
			       +deriv1minus+deriv2minus);
    }

    for(k=0;k<=PARTITION;k++) {
      rhoplus[k]+=deltarhoplus[k];
      rhominus[k]+=deltarhominus[k];
      if(rhoplus[k]<0.0) rhoplus[k]=0.0;
      if(rhominus[k]<0.0) rhominus[k]=0.0;
    }

    test=0.0;
    for(l=0;l<(PARTITION/4);l++) {
      for(n=0;n<5;n++) {
	k=l*4+n;
	test+=weights[n]*(rhoplus[k]+rhominus[k]);
      }
    }
    test*=HM_MoP;

    for(k=0;k<=PARTITION;k++) {
      rhoplus[k]/=test;
      rhominus[k]/=test;
    }

/*************************************************************************/

  }                                                  /*** sweeps loop ****/
  while(time<=10.0);

  q=saddle(q);
  fp=fopen(distfile,"a");
  fprintf(fp,"t %f\n",time);
  fprintf(fp,"h rhoplus rhominus \n");
  for(k=0;k<=PARTITION;k++) {
    fprintf(fp,"%f %f %f\n",HMIN+k*HM_MoP,rhoplus[k],rhominus[k]);
  }
  fclose(fp);
}                                                    /******* main *******/
/*************************************************************************/



/*************************************************************************/
/******************************** subroutines ****************************/
/*************************************************************************/



/****************************** allocate 3tensor *************************/

double ***f3tensor(nrl,nrh,ncl,nch,ndl,ndh)
     int nrl,nrh,ncl,nch,ndl,ndh;
{
  int i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
  double ***t;

  t=(double ***) malloc((unsigned)((nrow+NR_END)*sizeof(double**)));
  if (!t) nrerror("allocation failure 1 in f3tensor()");
  t+=NR_END;    t-=nrl;

  t[nrl]=(double **) malloc((unsigned)((nrow*ncol+NR_END)*sizeof(double*)));
  if (!t[nrl]) nrerror("allocation failure 2 in f3tensor()");
  t[nrl]+=NR_END;    t[nrl]-=ncl;

  t[nrl][ncl]=(double *) malloc((unsigned)((nrow*ncol*ndep+NR_END)*sizeof(double)));
  if (!t[nrl][ncl]) nrerror("allocation failure 3 in f3tensor()");
  t[nrl][ncl]+=NR_END;    t[nrl][ncl]-=ndl;

  for(j=ncl+1;j<=nch;j++)   t[nrl][j]=t[nrl][j-1]+ndep;
  for(i=nrl+1;i<=nrh;i++) {
    t[i]=t[i-1]+ncol;       t[i][ncl]=t[i-1][ncl]+ncol*ndep;
    for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
  }
  return t;
}
/*************************************************************************/



/****************************** calculate stuff **************************/

void calcstuff(q)
     double q;
{
  int i,j,k,ip,ipIp1,ip1pIp1,jp,jpIp1,jp1pIp1,l,n;
  double hJm,istep,jstep,explus[4],exminus[4];
  double sqrtq,lambda_1,Tlambda01,thetaoverr,rtl,rtlsqrt;
  double denominatorplus[2*INTERVAL+2][2*INTERVAL+2];
  double denominatorminus[2*INTERVAL+2][2*INTERVAL+2];

/*  printf("calcstuff %f\n",q);*/

  sqrtq=sqrt(q);
  Tlambda01=2*Js*(1.0-q);
  lambda_1=sqrtq/(J_1*(1.0-q));
  if(r!=0.0)
    thetaoverr=OK*Theta_1/r;
  else
    thetaoverr=0.0;
  rtl=lambda_1*lambda_1-thetaoverr*thetaoverr;
  if(rtl>0.0) rtlsqrt=sqrt(lambda_1*lambda_1-thetaoverr*thetaoverr);
  else rtlsqrt=lambda_1;


  for(i=0;i<=INTERVAL;i++) {
    ipIp1=i+INTERVAL+1;
    istep=-1.0+2.0*(double)i/(double)INTERVAL;

    for(j=0;j<=INTERVAL;j++) {
      jpIp1=j+INTERVAL+1;
      jstep=-1.0+2.0*(double)j/(double)INTERVAL;

/************************** calculate denominator ************************/

      denominatorplus[i][j]=0.0;
      denominatorplus[ipIp1][j]=0.0;
      denominatorplus[i][jpIp1]=0.0;
      denominatorplus[ipIp1][jpIp1]=0.0;

      denominatorminus[i][j]=0.0;
      denominatorminus[ipIp1][j]=0.0;
      denominatorminus[i][jpIp1]=0.0;
      denominatorminus[ipIp1][jpIp1]=0.0;

      for(l=0;l<(PARTITION/4);l++) {
	for(n=0;n<5;n++) {
	  k=l*4+n;
	  hJm=(HMIN+k*HM_MoP)-J_0*m;
	  
	  explus[0]=exp(-hJm*(hJm/Tlambda01-OK*Theta_0-thetaoverr*jstep)+r*jstep);
	  explus[1]=exp(mu+hJm*rtlsqrt*istep);
	  explus[2]=exp(-hJm*(hJm/Tlambda01-OK*Theta_0-thetaoverr/jstep)+r/jstep);
	  explus[3]=exp(mu+hJm*rtlsqrt/istep);
	  if(explus[0]>1.0E300) explus[0]=1.0E300;
	  if(explus[1]>1.0E300) explus[1]=1.0E300;
	  if(explus[2]>1.0E300) explus[2]=1.0E300;
	  if(explus[3]>1.0E300) explus[3]=1.0E300;

	  exminus[0]=exp(-hJm*(hJm/Tlambda01+OK*Theta_0-thetaoverr*jstep)-r*jstep);
	  exminus[1]=exp(-mu+hJm*rtlsqrt*istep);
	  exminus[2]=exp(-hJm*(hJm/Tlambda01+OK*Theta_0-thetaoverr/jstep)-r/jstep);
	  exminus[3]=exp(-mu+hJm*rtlsqrt/istep);
	  if(exminus[0]>1.0E300) exminus[0]=1.0E300;
	  if(exminus[1]>1.0E300) exminus[1]=1.0E300;
	  if(exminus[2]>1.0E300) exminus[2]=1.0E300;
	  if(exminus[3]>1.0E300) exminus[3]=1.0E300;

	  denominatorplus[i][j]+=weights[n]*chiplus[k]*explus[0]*explus[1];
	  denominatorplus[ipIp1][j]+=weights[n]*chiplus[k]*explus[0]*explus[3];
	  denominatorplus[i][jpIp1]+=weights[n]*chiplus[k]*explus[2]*explus[1];
	  denominatorplus[ipIp1][jpIp1]+=weights[n]*chiplus[k]*explus[2]*explus[3];

	  denominatorminus[i][j]+=weights[n]*chiminus[k]*exminus[0]*exminus[1];
	  denominatorminus[ipIp1][j]+=weights[n]*chiminus[k]*exminus[0]*exminus[3];
	  denominatorminus[i][jpIp1]+=weights[n]*chiminus[k]*exminus[2]*exminus[1];
	  denominatorminus[ipIp1][jpIp1]+=weights[n]*chiminus[k]*exminus[2]*exminus[3];
	}
      }                                              /****** l loop ******/

      denominatorplus[i][j]*=HM_MoP;
      denominatorplus[ipIp1][j]*=HM_MoP;
      denominatorplus[i][jpIp1]*=HM_MoP;
      denominatorplus[ipIp1][jpIp1]*=HM_MoP;

      denominatorminus[i][j]*=HM_MoP;
      denominatorminus[ipIp1][j]*=HM_MoP;
      denominatorminus[i][jpIp1]*=HM_MoP;
      denominatorminus[ipIp1][jpIp1]*=HM_MoP;

/************************** calculate measure ****************************/

      for(k=0;k<=PARTITION;k++) {
	hJm=(HMIN+k*HM_MoP)-J_0*m;

	explus[0]=exp(-hJm*(hJm/Tlambda01-OK*Theta_0-thetaoverr*jstep)+r*jstep);
	explus[1]=exp(mu+hJm*rtlsqrt*istep);
	explus[2]=exp(-hJm*(hJm/Tlambda01-OK*Theta_0-thetaoverr/jstep)+r/jstep);
	explus[3]=exp(mu+hJm*rtlsqrt/istep);
	if(explus[0]>1.0E300) explus[0]=1.0E300;
	if(explus[1]>1.0E300) explus[1]=1.0E300;
	if(explus[2]>1.0E300) explus[2]=1.0E300;
	if(explus[3]>1.0E300) explus[3]=1.0E300;

	exminus[0]=exp(-hJm*(hJm/Tlambda01+OK*Theta_0-thetaoverr*jstep)-r*jstep);
	exminus[1]=exp(-mu+hJm*rtlsqrt*istep);
	exminus[2]=exp(-hJm*(hJm/Tlambda01+OK*Theta_0-thetaoverr/jstep)-r/jstep);
	exminus[3]=exp(-mu+hJm*rtlsqrt/istep);
	if(exminus[0]>1.0E300) exminus[0]=1.0E300;
	if(exminus[1]>1.0E300) exminus[1]=1.0E300;
	if(exminus[2]>1.0E300) exminus[2]=1.0E300;
	if(exminus[3]>1.0E300) exminus[3]=1.0E300;

	measureplus[i][j][k]=chiplus[k]*explus[0]*explus[1]/
	  (denominatorplus[i][j]+denominatorminus[i][j]);
	measureplus[ipIp1][j][k]=chiplus[k]*explus[0]*explus[3]/
	  (denominatorplus[ipIp1][j]+denominatorminus[ipIp1][j]);
	measureplus[i][jpIp1][k]=chiplus[k]*explus[2]*explus[1]/
	  (denominatorplus[i][jpIp1]+denominatorminus[i][jpIp1]);
	measureplus[ipIp1][jpIp1][k]=chiplus[k]*explus[2]*explus[3]/
	  (denominatorplus[ipIp1][jpIp1]+denominatorminus[ipIp1][jpIp1]);

	measureminus[i][j][k]=chiminus[k]*exminus[0]*exminus[1]/
	  (denominatorplus[i][j]+denominatorminus[i][j]);
	measureminus[ipIp1][j][k]=chiminus[k]*exminus[0]*exminus[3]/
	  (denominatorplus[ipIp1][j]+denominatorminus[ipIp1][j]);
	measureminus[i][jpIp1][k]=chiminus[k]*exminus[2]*exminus[1]/
	  (denominatorplus[i][jpIp1]+denominatorminus[i][jpIp1]);
	measureminus[ipIp1][jpIp1][k]=chiminus[k]*exminus[2]*exminus[3]/
	  (denominatorplus[ipIp1][jpIp1]+denominatorminus[ipIp1][jpIp1]);
      }                                              /****** k loop ******/
    }                                                /****** j loop ******/
  }                                                  /****** i loop ******/

  check();
  calcrho();
}
/*************************************************************************/


/************************** check \int \sum rho = 1 **********************/

void check()
{
  int i,j,k,ipIp1,jpIp1,l,n,g,h;
  double w,y,ww,yy,ex1,ex3;
  double checkplus[2*INTERVAL+2][2*INTERVAL+2];
  double checkminus[2*INTERVAL+2][2*INTERVAL+2];
  double checkintermp[2*INTERVAL+2],checkintermm[2*INTERVAL+2];
  double chk,chkp,chkm;

/*  printf("check \n");*/

  for(i=0;i<=INTERVAL;i++) {
    ipIp1=i+INTERVAL+1;

    for(j=0;j<=INTERVAL;j++) {
      jpIp1=j+INTERVAL+1;

      checkplus[i][j]=0.0;            checkplus[ipIp1][j]=0.0;
      checkplus[i][jpIp1]=0.0;        checkplus[ipIp1][jpIp1]=0.0;
      checkminus[i][j]=0.0;           checkminus[ipIp1][j]=0.0;
      checkminus[i][jpIp1]=0.0;       checkminus[ipIp1][jpIp1]=0.0;

      for(l=0;l<(PARTITION/4);l++) {
	for(n=0;n<5;n++) {
	  k=l*4+n;
	  checkplus[i][j]+=weights[n]*measureplus[i][j][k];
	  checkplus[ipIp1][j]+=weights[n]*measureplus[ipIp1][j][k];
	  checkplus[i][jpIp1]+=weights[n]*measureplus[i][jpIp1][k];
	  checkplus[ipIp1][jpIp1]+=weights[n]*measureplus[ipIp1][jpIp1][k];
	  
	  checkminus[i][j]+=weights[n]*measureminus[i][j][k];
	  checkminus[ipIp1][j]+=weights[n]*measureminus[ipIp1][j][k];
	  checkminus[i][jpIp1]+=weights[n]*measureminus[i][jpIp1][k];
	  checkminus[ipIp1][jpIp1]+=weights[n]*measureminus[ipIp1][jpIp1][k];
	}
      }
      checkplus[i][j]*=HM_MoP;            checkplus[ipIp1][j]*=HM_MoP;
      checkplus[i][jpIp1]*=HM_MoP;        checkplus[ipIp1][jpIp1]*=HM_MoP;
      checkminus[i][j]*=HM_MoP;           checkminus[ipIp1][j]*=HM_MoP;
      checkminus[i][jpIp1]*=HM_MoP;       checkminus[ipIp1][jpIp1]*=HM_MoP;
    }
  }
  

  for(i=0;i<=INTERVAL;i++) {
    ipIp1=i+INTERVAL+1;
    checkintermp[i]=0.0;                          checkintermp[ipIp1]=0.0;
    checkintermm[i]=0.0;                          checkintermm[ipIp1]=0.0;

    for(g=0;g<(INTERVAL/(ISTEP-1));g++) {
      for(h=0;h<ISTEP;h++) {
	j=g*(ISTEP-1)+h;
	jpIp1=j+INTERVAL+1;
	w=-1.0+2.0*(double)j/(double)INTERVAL;      ww=w*w;
	ex1=exp(-0.5*ww);                           ex3=exp(-0.5/ww)/ww;

	checkintermp[i]+=sweight[h]*(ex1*checkplus[i][j]+ex3*checkplus[i][jpIp1]);
	checkintermp[ipIp1]+=sweight[h]*(ex1*checkplus[ipIp1][j]+ex3*checkplus[ipIp1][jpIp1]);

	checkintermm[i]+=sweight[h]*(ex1*checkminus[i][j]+ex3*checkminus[i][jpIp1]);
	checkintermm[ipIp1]+=sweight[h]*(ex1*checkminus[ipIp1][j]+ex3*checkminus[ipIp1][jpIp1]);
      }
    }
  }
  
  chkp=0.0;
  chkm=0.0;

  for(g=0;g<(INTERVAL/(ISTEP-1));g++) {
    for(h=0;h<ISTEP;h++) {
      i=g*(ISTEP-1)+h;
      ipIp1=i+INTERVAL+1;
      y=-1.0+2.0*(double)i/(double)INTERVAL;       yy=y*y;
      ex1=exp(-0.5*yy);                            ex3=exp(-0.5/yy)/yy;

      chkp+=sweight[h]*(ex1*checkintermp[i]+ex3*checkintermp[ipIp1]);
      chkm+=sweight[h]*(ex1*checkintermm[i]+ex3*checkintermm[ipIp1]);
    }
  }

  chkp*=TWOoPI/INTSQUARE;
  chkm*=TWOoPI/INTSQUARE;
  chk=chkp+chkm;

  if(fabs(chk-1.0)>0.01)
    printf("check error one %f plus %f minus %f\n",chk,chkp,chkm);

  for(i=0;i<=INTERVAL;i++) {
    ipIp1=i+INTERVAL+1;
    for(j=0;j<=INTERVAL;j++) {
      jpIp1=j+INTERVAL+1;
      for(k=0;k<=PARTITION;k++) {
	measureplus[i][j][k]/=chk;         measureminus[i][j][k]/=chk;
	measureplus[i][jpIp1][k]/=chk;     measureminus[i][jpIp1][k]/=chk;
	measureplus[ipIp1][j][k]/=chk;     measureminus[ipIp1][j][k]/=chk;
	measureplus[ipIp1][jpIp1][k]/=chk; measureminus[ipIp1][jpIp1][k]/=chk;
      }
    }
  }
}
/*************************************************************************/


/****************** solve saddle point equations *************************/

double saddle(q)
     double q;
{
  int k,l,n,ti,i;
  double oldq,dq,doq,rp,hJm,oldr,oldt0,oldt1;
  double oldtterm,oldrterm,oldoldq,step;
  double oldchiplus[PARTITION+1],oldchiminus[PARTITION+1];


  for(k=0;k<=PARTITION;k++) {
    oldchiplus[k]=chiplus[k];
    oldchiminus[k]=chiminus[k];
  }
  oldtterm=tterm;            oldrterm=rterm;  oldoldq=q;

  m=0.0;       newmu=0.0;       newr1=0.0;       newt1=0.0;

  for(l=0;l<(PARTITION/4);l++) {
    for(n=0;n<5;n++) {
      k=l*4+n;
      m+=weights[n]*(rhoplus[k]-rhominus[k]);
    }
  }
  m*=HM_MoP;

  for(l=0;l<(PARTITION/4);l++) {
    for(n=0;n<5;n++) {
      k=l*4+n;
      hJm=(HMIN+k*HM_MoP)-J_0*m;
      newmu+=weights[n]*hJm*(rhoplus[k]+rhominus[k]);
      newr1+=weights[n]*hJm*hJm*(rhoplus[k]+rhominus[k]);
      newt1+=weights[n]*hJm*(rhoplus[k]-rhominus[k]);
    }
  }
  newmu*=HM_MoP;                 newr1*=HM_MoP;            newt1*=HM_MoP;

  calcmmur(q);


  for(step=3.0;step>=1.0;step-=1.0) {
    ti=0;
/*    printf("step %f\n",step);*/
    do {
      oldq=q;

      calcstuff(q);  calct();
      rp=0.0;
      for(k=0;k<=PARTITION;k++) {
	chiplus[k]-=step*GRAD*(calcrhoplus[k]-rhoplus[k])/chiplus[k];
	chiminus[k]-=step*GRAD*(calcrhominus[k]-rhominus[k])/chiminus[k];

	rp+=(rhoplus[k]-calcrhoplus[k])*(rhoplus[k]-calcrhoplus[k]);
	rp+=(rhominus[k]-calcrhominus[k])*(rhominus[k]-calcrhominus[k]);
      }
      rp/=2.0*(PARTITION+1.0);
      rp=sqrt(rp);
/*      printf("rp %f\n",rp);*/


      calcmmur(q);    calcstuff(q);    calct();      q=calcq(q);
      calcmmur(q);

      if(oldq>0.0001)
	dq=(q-oldq)/oldq;
      else dq=(q-oldq);


/*      printf("m %f mu %f r %f T0 %f T1 %f q %f dq %f\n",m,mu,r,Theta_0,Theta_1,q,dq);*/

      if(q<0.0 || q >1.0 || fabs(dq)>0.1) {
	for(k=0;k<=PARTITION;k++) {
	  chiplus[k]=oldchiplus[k];
	  chiminus[k]=oldchiminus[k];
	}
	tterm=oldtterm;            rterm=oldrterm;  q=oldoldq;
	break;
      }
      ti++;
    }
    while(fabs(dq)>step*step*ACCURACY || rp>RTEST && ti<=NSTEPS);
  }

  return(q);
}
/*************************************************************************/


/******************* calculate saddle point equations ********************/

void calcmmur(q)
     double q;
{
  double omq,omqs,rsquared,oldr;

  oldr=r;
  omq=1.0-q;
  omqs=Js*(1.0-q)*(1.0-q);

  mu=J_0/(Js*omq)*newmu;
  Theta_0=1.0/(2.0*omqs)*(newt1-tterm);
  mu-=OK*J_0*m*Theta_0;
  Theta_1=-1.0/(2.0*omqs)*(2.0*Js*(1.0-q*q)*Theta_0-newt1);
  rsquared=Js*q+(-2.0*q*newr1+(1.0+q)*rterm)/omq
    -2.0*Js*OK*Theta_0*(-3.0*q*newt1+(2.0+q)*tterm);

  if(rsquared>=0.0) r=sqrt(rsquared)/(J_1*omq);
  else r=oldr;

/*  printf("nr1 %f rterm %f t0 newt1 %f tterm %f rsquared %f\n",newr1,rterm,newt1,tterm,rsquared);*/


/*  printf("q %f m %f mu %f r %f Theta_0 %f Theta_1 %f\n",q,m,mu,r,Theta_0,Theta_1);*/
}
/*************************************************************************/


/**************************** calculate q ********************************/

double calcq(q)
     double q;
{
  int k,l,n;

/*  printf("q %f\n",q);*/

  q=0.0;  
  for(l=0;l<(PARTITION/4);l++) {
    for(n=0;n<5;n++) {
      k=l*4+n;
      q+=weights[n]*(rhoplussr[k]-rhominussr[k]);
    }
  }
  q*=HM_MoP;

  return(q);
}
/*************************************************************************/


/*************************** evaluate coefficients ***********************/

void calccoefficients_wy(q,time,flowfile)
     double q,time;
     char flowfile[10];
{
  FILE *fp;
  int l,n,k;
  double hJm,th,at;

/*  printf("coeffs %f\n",q);*/

  calcstuff(q);

  tanhplus=0.0;
  tanhminus=0.0;
  tanhsigmarho=0.0;
  htanh=0.0;
  h_tanh=0.0;
  E=0.0;

  for(l=0;l<(PARTITION/4);l++) {
    for(n=0;n<5;n++) {
      k=l*4+n;
      hJm=(HMIN+k*HM_MoP)-J_0*m;

      if(T!=0.0) th=tanh(beta*(HMIN+k*HM_MoP));
      else {
	if(HMIN+k*HM_MoP>0.0)  th=1.0;
	if(HMIN+k*HM_MoP<0.0)  th=-1.0;
	if(HMIN+k*HM_MoP==0)   th=0.0;
      }

      tanhplus+=weights[n]*th*rhoplus[k];
      tanhminus+=weights[n]*th*rhominus[k];
      tanhsigmarho+=weights[n]*th*(rhoplussr[k]+rhominussr[k]);
      htanh+=weights[n]*hJm*th*(rhoplus[k]+rhominus[k]);
      h_tanh+=weights[n]*th*(rhoplush[k]+rhominush[k]);
      E+=weights[n]*(HMIN+k*HM_MoP)*(rhoplus[k]-rhominus[k]);
    }
  }

  tanhplus*=HM_MoP;
  tanhminus*=HM_MoP;
  tanhsigmarho*=HM_MoP;
  htanh*=HM_MoP;
  h_tanh*=HM_MoP;
  E*=-0.5*HM_MoP;

/*  printf("coefs %f %f %f %f %f\n",tanhplus,tanhminus,tanhsigmarho,htanh,h_tanh);*/

/*  at=calcAT(q);*/
at=0.0;

/*************************** EXPORT  dm/dt dE/dt *************************/

/*  printf("tanh %f %f %f m %f E %f\n",tanhplus,tanhminus,htanh,m,E);*/

  fp=fopen(flowfile,"a");
  fprintf(fp,"%f %f %f %f %f %f\n",(float)time,(float)m,(float)E,(float)(tanhplus+tanhminus-m),
	  (float)(-htanh-J_0*m*(tanhplus+tanhminus)-2.0*E),q);
  fclose(fp);
}
/*************************************************************************/


/***************************** calculate rho *****************************/

void calcrho()     
{
  int i,j,k,kprime,l,n,ipIp1,jpIp1,g,hi;
  double y,yy,w,ww,ex1,ex3,hJm,test,oldrterm,oldtterm;
  double intermplus[2*INTERVAL+2],intermminus[2*INTERVAL+2];
  double intermpp[2*INTERVAL+2],intermpm[2*INTERVAL+2];
  double intermmp[2*INTERVAL+2],intermmm[2*INTERVAL+2];
  double rhopp[PARTITION+1][PARTITION+1],rhopm[PARTITION+1][PARTITION+1];
  double rhomp[PARTITION+1][PARTITION+1],rhomm[PARTITION+1][PARTITION+1];


/*  printf("rho \n");*/

  for(k=0;k<=PARTITION;k++) {

/************************** calculate rhorho *****************************/

    for(kprime=0;kprime<=PARTITION;kprime++) {
      rhopp[k][kprime]=0.0;                        rhopm[k][kprime]=0.0;
      rhomp[k][kprime]=0.0;                        rhomm[k][kprime]=0.0;

      for(i=0;i<=INTERVAL;i++) {
	ipIp1=i+INTERVAL+1;
	intermpp[i]=0.0;                           intermpp[ipIp1]=0.0;
	intermpm[i]=0.0;                           intermpm[ipIp1]=0.0;
	intermmp[i]=0.0;                           intermmp[ipIp1]=0.0;
	intermmm[i]=0.0;                           intermmm[ipIp1]=0.0;

	for(g=0;g<(INTERVAL/(ISTEP-1));g++) {
	  for(hi=0;hi<ISTEP;hi++) {
	    j=g*(ISTEP-1)+hi;
	    jpIp1=j+INTERVAL+1;
	    w=-1.0+2.0*(double)j/(double)INTERVAL; ww=w*w;
	    ex1=exp(-0.5*ww);                      ex3=exp(-0.5/ww)/ww;

	    intermpp[i]+=sweight[hi]*(ex1*measureplus[i][j][k]*measureplus[i][j][kprime]
				      +ex3*measureplus[i][jpIp1][k]*measureplus[i][jpIp1][kprime]);
	    intermpp[ipIp1]+=sweight[hi]*(ex1*measureplus[ipIp1][j][k]*measureplus[ipIp1][j][kprime]
					  +ex3*measureplus[ipIp1][jpIp1][k]*measureplus[ipIp1][jpIp1][kprime]);

	    intermpm[i]+=sweight[hi]*(ex1*measureplus[i][j][k]*measureminus[i][j][kprime]
				      +ex3*measureplus[i][jpIp1][k]*measureminus[i][jpIp1][kprime]);
	    intermpm[ipIp1]+=sweight[hi]*(ex1*measureplus[ipIp1][j][k]*measureminus[ipIp1][j][kprime]
					  +ex3*measureplus[ipIp1][jpIp1][k]*measureminus[ipIp1][jpIp1][kprime]);

	    intermmp[i]+=sweight[hi]*(ex1*measureminus[i][j][k]*measureplus[i][j][kprime]
				      +ex3*measureminus[i][jpIp1][k]*measureplus[i][jpIp1][kprime]);
	    intermmp[ipIp1]+=sweight[hi]*(ex1*measureminus[ipIp1][j][k]*measureplus[ipIp1][j][kprime]
					  +ex3*measureminus[ipIp1][jpIp1][k]*measureplus[ipIp1][jpIp1][kprime]);
	    
	    intermmm[i]+=sweight[hi]*(ex1*measureminus[i][j][k]*measureminus[i][j][kprime]
				      +ex3*measureminus[i][jpIp1][k]*measureminus[i][jpIp1][kprime]);
	    intermmm[ipIp1]+=sweight[hi]*(ex1*measureminus[ipIp1][j][k]*measureminus[ipIp1][j][kprime]
					  +ex3*measureminus[ipIp1][jpIp1][k]*measureminus[ipIp1][jpIp1][kprime]);
	  }
	}
      }

      for(g=0;g<(INTERVAL/(ISTEP-1));g++) {
	for(hi=0;hi<ISTEP;hi++) {
	  i=g*(ISTEP-1)+hi;
	  ipIp1=i+INTERVAL+1;
	  y=-1.0+2.0*(double)i/(double)INTERVAL;     yy=y*y;
	  ex1=exp(-0.5*yy);                          ex3=exp(-0.5/yy)/yy;
	  
	  rhopp[k][kprime]+=sweight[hi]*(ex1*intermpp[i]+ex3*intermpp[ipIp1]);
	  rhopm[k][kprime]+=sweight[hi]*(ex1*intermpm[i]+ex3*intermpm[ipIp1]);
	  rhomp[k][kprime]+=sweight[hi]*(ex1*intermmp[i]+ex3*intermmp[ipIp1]);
	  rhomm[k][kprime]+=sweight[hi]*(ex1*intermmm[i]+ex3*intermmm[ipIp1]);
	}
      }
      rhopp[k][kprime]*=TWOoPI/INTSQUARE;
      rhopm[k][kprime]*=TWOoPI/INTSQUARE;
      rhomp[k][kprime]*=TWOoPI/INTSQUARE;
      rhomm[k][kprime]*=TWOoPI/INTSQUARE;
    }                                                /*** kprime loop ****/


/************ calculate rhoplus-rhominus and h*(rhoplus+rhominus) ********/

    rhoplussr[k]=0.0;               rhominussr[k]=0.0;
    rhoplush[k]=0.0;                rhominush[k]=0.0;

    for(l=0;l<(PARTITION/4);l++) {
      for(n=0;n<5;n++) {
	kprime=l*4+n;
	hJm=(HMIN+kprime*HM_MoP)-J_0*m;
	
	rhoplussr[k]+=weights[n]*(rhopp[k][kprime]-rhopm[k][kprime]);
	rhominussr[k]+=weights[n]*(rhomp[k][kprime]-rhomm[k][kprime]);
	
	rhoplush[k]+=weights[n]*hJm*(rhopp[k][kprime]+rhopm[k][kprime]);
	rhominush[k]+=weights[n]*hJm*(rhomp[k][kprime]+rhomm[k][kprime]);
      }
    }
    rhoplussr[k]*=HM_MoP;           rhominussr[k]*=HM_MoP;
    rhoplush[k]*=HM_MoP;            rhominush[k]*=HM_MoP;


/************************ calculate rhoplus and rhominus *****************/

    calcrhoplus[k]=0.0;                  calcrhominus[k]=0.0;
    for(i=0;i<=INTERVAL;i++) {
      ipIp1=i+INTERVAL+1;
      intermplus[i]=0.0;             intermplus[ipIp1]=0.0;
      intermminus[i]=0.0;            intermminus[ipIp1]=0.0;

      for(g=0;g<(INTERVAL/(ISTEP-1));g++) {
	for(hi=0;hi<ISTEP;hi++) {
	  j=g*(ISTEP-1)+hi;
	  jpIp1=j+INTERVAL+1;
	  w=-1.0+2.0*(double)j/(double)INTERVAL;      ww=w*w;
	  ex1=exp(-0.5*ww);                           ex3=exp(-0.5/ww)/ww;

	  intermplus[i]+=sweight[hi]*(ex1*measureplus[i][j][k]
				      +ex3*measureplus[i][jpIp1][k]);
	  intermplus[ipIp1]+=sweight[hi]*(ex1*measureplus[ipIp1][j][k]
					  +ex3*measureplus[ipIp1][jpIp1][k]);
	  
	  intermminus[i]+=sweight[hi]*(ex1*measureminus[i][j][k]
				       +ex3*measureminus[i][jpIp1][k]);
	  intermminus[ipIp1]+=sweight[hi]*(ex1*measureminus[ipIp1][j][k]
					   +ex3*measureminus[ipIp1][jpIp1][k]);
	}
      }
    }

    for(g=0;g<(INTERVAL/(ISTEP-1));g++) {
      for(hi=0;hi<ISTEP;hi++) {
	i=g*(ISTEP-1)+hi;
	ipIp1=i+INTERVAL+1;
	y=-1.0+2.0*(double)i/(double)INTERVAL;       yy=y*y;
	ex1=exp(-0.5*yy);                            ex3=exp(-0.5/yy)/yy;
      
	calcrhoplus[k]+=sweight[hi]*(ex1*intermplus[i]+ex3*intermplus[ipIp1]);
	calcrhominus[k]+=sweight[hi]*(ex1*intermminus[i]+ex3*intermminus[ipIp1]);
      }
    }
    calcrhoplus[k]*=TWOoPI/INTSQUARE;
    calcrhominus[k]*=TWOoPI/INTSQUARE;
  }                                                  /****** k loop ******/

  test=0.0;
  for(l=0;l<(PARTITION/4);l++) {
    for(n=0;n<5;n++) {
      k=l*4+n;
      test+=weights[n]*(calcrhoplus[k]+calcrhominus[k]);
    }
  }
  test*=HM_MoP;

  if(fabs(test-1.0)>0.05) printf("calcrho test error %f\n",test);

  for(k=0;k<=PARTITION;k++) {
    calcrhoplus[k]/=test;
    calcrhominus[k]/=test;
  }
}
/*************************************************************************/


/********** calculate non-trivial terms in saddle point equations ********/

void calct()
{
  int l,n,k;
  double hJm;
  
  rterm=0.0;                   tterm=0.0;

  for(l=0;l<(PARTITION/4);l++) {
    for(n=0;n<5;n++) {
      k=l*4+n;
      hJm=(HMIN+k*HM_MoP)-J_0*m;
      rterm+=weights[n]*hJm*(rhoplush[k]+rhominush[k]);
      tterm+=weights[n]*(rhoplush[k]-rhominush[k]);
    }
  }
  rterm*=HM_MoP;
  tterm*=HM_MoP;
}
/*************************************************************************/
