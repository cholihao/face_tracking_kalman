
//#include "/usr/local/Cellar/eigen/3.3.1/include/eigen3/Eigen/Dense" //Include Eigen Library
#include <Eigen/Dense>
using namespace Eigen;

class NNKalmanFilter {

	public:

		/* Problem Dimension */
		int n; //State vector dimension
		int m; //Control vector (input) dimension (if there is not input, set to zero)

		/* Fixed Matrix */
		MatrixXf A; //System dynamics matrix
		MatrixXf B; //Control matrix
		MatrixXf H; //Mesaurement Adaptation matrix
		MatrixXf Q; //Process Noise Covariance matrix
		MatrixXf R; //Measurement Noise Covariance matrix
		MatrixXf I; //Identity matrix

		/* Variable Matrix */
		VectorXf X; //(Current) State vector
		MatrixXf P; //State Covariance
		MatrixXf K; //Kalman Gain matrix

		/* Inizial Value */
		VectorXf X0; //Initial State vector
		MatrixXf P0; //Initial State Covariance matrix

		NNKalmanFilter();
		/*
		* Constructor
		* _n: state vector dimension
		* _m: control vector dimension (if there is not input, set to zero)
		*/
		NNKalmanFilter(int _n,  int _m);

		/* Set Fixed Matrix (NO INPUT) */
		//void setFixed ( MatrixXf _A, MatrixXf _H, MatrixXf _Q, MatrixXf _R );
		void setFixed( MatrixXf _X, MatrixXf _A, MatrixXf _H, MatrixXf _Q, MatrixXf _R, MatrixXf P, MatrixXf _K);

		/* Set Fixed Matrix (WITH INPUT) */
		void setFixed ( MatrixXf _A, MatrixXf _B, MatrixXf _H, MatrixXf _Q, MatrixXf _R );

		/* Set Initial Value */
		void setInitial( VectorXf _X0, MatrixXf _P0 );

		/* Do prediction (NO INPUT) */
		void predict ( void );

		/* Do prediction (INPUT) */
		void predict ( VectorXf U );

		/* Do correction */
		void correct ( VectorXf &Z );

};





