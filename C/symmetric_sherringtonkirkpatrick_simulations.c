/*************************************************************************/
/**************  simulations for asymmetric s-k model  *******************/
/*************************************************************************/
/************************ stephen laughton 9/11/94 ***********************/
/*************************************************************************/
/******************  DEC alpha cc code.c -lm -O  *************************/
/*************************************************************************/


#define PI 3.1415926526

#include <stdio.h>
#include <math.h>
#include "extra/random.h"
#include "extra/extmath.h"
#include "extra/matrices.h"

#define SPINS 8000
#define ITERATIONS (SPINS*10)
#define K 0.0

void EXPORT(), DISTRIBUTION();
float SJ();
float J0,J,T,**z,old[2];


main()
{
  FILE *fp;
  int i,j,k,site;
  int flip,flips,initial,iter;
  int indices[SPINS],index;
  int *s;
  float betaoverN,roottwo,rootN,roottwoN,field;
  float SYMFIELD();
  float **matrix();
  double RANDOM(),erfinv();

  char flowfile[12],noisefile[12];

  s=ivector(1,SPINS);
  z=matrix(1,(SPINS/2),1,(SPINS-1));

  old[0]=0.0;
  old[1]=0.0;
  J=1.0;
  roottwo=sqrt(2.0);
  rootN=sqrt((float)SPINS);
  roottwoN=roottwo*rootN;

  SETRANDOM();
  for(i=1;i<=(int)(SPINS/2);i++) {          /**** randomise z[i][j] ****/
    for(j=1;j<=SPINS-1;j++) {
      z[i][j]=roottwoN*erfinv(2.0*RANDOM()-1.0);
    }
  }

  for(T=0.0;T<=1.0;T+=1.0) {
    if(T!=0.0) betaoverN=1.0/(T*(float)(SPINS));
    for(J0=0.0;J0<=2.0;J0+=1.0) {

      for(initial=0;initial<10;initial+=3) {

	sprintf(flowfile,"mande.%d.%d.%d",(int)T,(int)J0,initial);
	sprintf(noisefile,"noise.%d.%d.%d",(int)T,(int)J0,initial);
	fp=fopen(flowfile,"w");
	fprintf(fp,"SPINS= %d\n",SPINS);
	fprintf(fp,"T= %f\n",T);
	fprintf(fp,"J0= %f\n",J0);
	fprintf(fp,"initial= %d\n",initial);
	fprintf(fp,"t   m   E   dm/dt  dE/dt\n");
	fclose(fp);
	
	fp=fopen(noisefile,"w");
	fprintf(fp,"SPINS= %d\n",SPINS);
	fprintf(fp,"T= %f\n",T);
	fprintf(fp,"J0= %f\n",J0);
	fprintf(fp,"initial= %d\n",initial);
	fclose(fp);

	for(i=1;i<=SPINS;i++) {             /****** sets spins to 1 ******/
	  s[i]=1;
	  indices[i]=i;
	}

	flips=(int)(0.5+0.05*(10-initial)*SPINS);
	for(i=0;i<flips;i++) {              /** sends index of flipped ***/
	  j=1+(int)(RANDOM()*(SPINS-i));    /** to end of array so no  ***/
	  index=indices[j];                 /** spin is flipped twice  ***/

	  if(j<(SPINS-i)) {
	    for(k=j+1;k<=(SPINS-i);k++) {
	      indices[k-1]=indices[k];
	    }
	  }
	  s[index]=-1;                      /******** flips spin *********/
	}                                   /******** i loop  ************/

	iter=0;
	EXPORT(s,iter,flowfile);   /*** symmetric interaction ***/
	DISTRIBUTION(s,iter,noisefile);

	for(iter=1;iter<ITERATIONS;iter++) {
	  site=(int)(RANDOM()*SPINS)+1;
	  field=SYMFIELD(site,s);      /** symmetric interaction ****/
	  if(s[site]<0)
	    field=-1.0*field;

	  if(T!=0)                          /***** transition rates ******/
	    flip=((2.0*RANDOM())<1.0-tanh(betaoverN*field));
	  else
	    flip=(field<0.0);
	  if(flip) {                        /********* flip spin *********/
	    if(s[site]<0)
	      s[site]=1;
	    else
	      s[site]=-1;
	  }

	  if(iter%(SPINS/40)==0) {
	    EXPORT(s,iter,flowfile);       /* symmetric interact*/
	  }
	  if(iter%(SPINS*2)==0) {
	    DISTRIBUTION(s,iter,noisefile);  /** symmetric interac **/
	  }
	}                                   /***** iterations loop *******/
	EXPORT(s,iter,flowfile);
	DISTRIBUTION(s,iter,noisefile);      /*** sym interaction ***/

      }                                     /******** initial loop *******/
    }                                       /******** J loop  ************/
  }                                         /******** T loop  ************/

}                                           /*********** main ************/

/*************************************************************************/
/******************************** subroutines ****************************/
/*************************************************************************/



/*************** find field at site for symmetric interactions ***********/


float SYMFIELD(site,s)
     int site,s[SPINS];
{
  int j;
  float field;
  
  field=0.0;
  if(site>1) {
    for(j=1;j<site;j++) {
      if(s[j]>0)
	field+=(J0+SJ(site,j));
      else
	field-=(J0+SJ(site,j));
    }
  }
  if(site<SPINS) {
    for(j=SPINS;j>site;j--) {
      if(s[j]>0)
	field+=(J0+SJ(site,j));
      else
	field-=(J0+SJ(site,j));
    }
  }

  return(field);
}

/*************************************************************************/




/*************************** export m and E values ***********************/

void EXPORT(s,iter,flowfile)    /* symmetric interaction */
     int s[SPINS],iter;
     char flowfile[12];
{
  FILE *fp;
  int i,j;
  float m,E,field;

  m=0;
  E=0;

  for(i=1;i<=SPINS;i++) {
    if(s[i]>0)
      m+=1;
    else
      m-=1;
    
    field=SYMFIELD(i,s);               /*** symmetric interaction ***/
    if(s[i]>0)
      E+=field;
    else
      E-=field;
  }

  m/=(float)SPINS;
  E/=(-2.0*(float)SPINS*(float)SPINS);
  

  fp=fopen(flowfile,"a");
  fprintf(fp,"%f %f  %f  %f  %f\n",(float)iter/(float)SPINS,(float)m,(float)E,(float)(m-old[0])*40.0,(float)(E-old[1])*40.0);
  fclose(fp);
  old[0]=m;
  old[1]=E;
}

/*************************************************************************/

/*********************** find field distribution *************************/

void DISTRIBUTION(s,iter,noisefile)             /* symmetric interaction */

     int s[SPINS],iter;
     char noisefile[12];

{
  FILE *fp;
  int i,j,array[2][(int)((float)SPINS/100.0)+1];
  float m,E,field;
  float minfield[2],maxfield[2];

  m=0;
  E=0;
  for(j=0;j<=(int)((float)SPINS/100.0);j++) {
    array[0][j]=0;
    array[1][j]=0;
  }

  minfield[0]=4.0;
  minfield[1]=4.0;
  maxfield[0]=-4.0;
  maxfield[1]=-4.0;

  for(i=1;i<=SPINS;i++) {
    field=SYMFIELD(i,s)/(float)SPINS;     /* symmetric interaction **/
    if(s[i]>0) {
      if(field<minfield[0])
	minfield[0]=field;
      if(field>maxfield[0])
	maxfield[0]=field;
    }
    else{
      if(field<minfield[1])
	minfield[1]=field;
      if(field>maxfield[1])
	maxfield[1]=field;
    }
  }

  for(i=1;i<=SPINS;i++) {
    if(s[i]>0)
      m+=1;
    else
      m-=1;

    field=SYMFIELD(i,s);               /*** symmetric interaction ***/
    field/=(float)SPINS;

    if(s[i]>0) {
      if(field<minfield[0]+0.5*(maxfield[0]-minfield[0])/((float)SPINS/100.0))
	array[0][0]+=1;
      else
	array[0][(int)((field-minfield[0])/(maxfield[0]-minfield[0])*((float)SPINS/100.0)+0.5)]+=1;
    }
    else {
      if(field<minfield[1]+0.5*(maxfield[1]-minfield[01])/((float)SPINS/100.0))
	array[1][0]+=1;
      else
	array[1][(int)((field-minfield[1])/(maxfield[1]-minfield[1])*((float)SPINS/100.0)+0.5)]+=1;
    }

    if(s[i]>0)
      E+=field;
    else
      E-=field;
  }

  m/=(float)SPINS;
  E/=(-2.0*(float)SPINS);
  
  
  fp=fopen(noisefile,"a");
  fprintf(fp,"m = %f  E = %f  time= %f\n",(float)m,(float)E,(float)iter/(float)SPINS);
  fprintf(fp,"h  rho+ (h)  h  rho- (h)\n");

  for(i=0;i<=(int)((float)SPINS/100.0);i++) {
    fprintf(fp,"%f  %f  %f  %f\n",(float)minfield[0]+i*(maxfield[0]-minfield[0])/((float)SPINS/100.0),(float)array[0][i]/(100.0*(maxfield[0]-minfield[0])),minfield[1]+i*(maxfield[1]-minfield[1])/((float)SPINS/100.0),(float)array[1][i]/(100.0*(maxfield[1]-minfield[1])));
  }
  fclose(fp);

}

/*************************************************************************/


/****************** return interactions J_{i j} **************************/

float SJ(i,j)
     int i,j;
{
  if(j<i) {
    if(i<=(SPINS/2))
      return(J*z[i][j]);
    else
      return(J*z[SPINS+1-i][SPINS-j]);
  }
  else
    return(SJ(j,i));
}

/*************************************************************************/




