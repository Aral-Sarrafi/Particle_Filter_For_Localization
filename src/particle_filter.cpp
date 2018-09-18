/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
// Define the random engine for random sampling from the distributions
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (unsigned int i = 0; i < num_particles; i++) {
		Particle p;

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.y = dist_theta(gen);

		p.id = i;

		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);

	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Define the noise distributions
	normal_distribution<double> n_x(0, std_pos[0]);
	normal_distribution<double> n_y(0, std_pos[1]);
	normal_distribution<double> n_theta(0, std_pos[2]);

	// Loop over all the particles and predict the new particles based on the CTRV model

	for (unsigned int i = 0; i < particles.size(); i++) {
		Particle p;
		p = particles[i];

		double x_p, y_p, theta_p;
		x_p = p.x;
		y_p = p.y;
		theta_p = p.theta;

		double new_x, new_y, new_theta;

		if (fabs(yaw_rate) < 0.00001) {

			new_x = x_p + velocity * cos(theta_p) * delta_t + n_x(gen);
			new_y = y_p + velocity * sin(theta_p) * delta_t + n_y(gen);
			new_theta = theta_p + n_theta(gen);

		}
		else
		{
			new_x = x_p + (velocity / yaw_rate) * (sin(theta_p + yaw_rate * delta_t) - sin(theta_p)) + n_x(gen);
			new_y = y_p + (velocity / yaw_rate) * (cos(theta_p) - cos(theta_p + yaw_rate * delta_t)) + n_y(gen);
			new_theta = theta_p + yaw_rate * delta_t + n_theta(gen);
		}
		// Update the predicted particles
		particles[i].x = new_x;
		particles[i].y = new_y;
		particles[i].theta = new_theta;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// Loop over all the obeservation and compare each of the to the predicted landmarks for find the best match.

	for (unsigned int i = 0; i < observations.size(); i++) {
		LandmarkObs obs;
		obs = observations[i];

		double x_obs, y_obs;
		x_obs = obs.x;
		y_obs = obs.y;

		double min_distance;
		min_distance = numeric_limits<double>::infinity();

		// Define a default ID for the cases where no match is found.
		int ID = -1;
		// Loop over all the predicted landmarks to find the best match for the selected observation.
		for (unsigned int j = 0; predicted.size(); j++) {
			LandmarkObs pr;
			pr = predicted[j];

			double x_pr, y_pr;
			x_pr = pr.x;
			y_pr = pr.y;

			double distance;
			distance = dist(x_obs, y_obs, x_pr, y_pr);

			if (distance < min_distance)
			{
				ID = pr.id;
				min_distance = distance;
			}

		}
		observations[i].id = ID;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// Loop over all the particles to update the weights.

	for (unsigned int i = 0; i < particles.size(); i++) {
		
		Particle p;
		p = particles[i];

		double x_p, y_p, theta_p;
		x_p = p.x;
		y_p = p.y;
		theta_p = p.theta;

		// For a spesific particle loop over all the landmarks to find which landmarks are in the range of sensor.
		// populate the predicted vector with the landmarks that are in the range of sensor.
		vector<LandmarkObs> predicted;
		double x_lm, y_lm;
		int ID_lm;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			
			x_lm = map_landmarks.landmark_list[j].x_f;
			y_lm = map_landmarks.landmark_list[j].y_f;
			ID_lm = map_landmarks.landmark_list[j].id_i;

			double distance_to_particle;
			distance_to_particle = dist(x_p, y_p, x_lm, y_lm);

			if (distance_to_particle <= sensor_range)
			{
				predicted.push_back(LandmarkObs{ x_lm, y_lm, ID_lm });
			}
		}

		// Loop over all the observations and transform them to the particle coordinate system.
		vector<LandmarkObs> Transformed_Obs;
		double x_obs, y_obs;
		int ID_obs;

		double x_tObs, y_tObs;
		for (unsigned int j = 0; j < observations.size(); j++) {
			
			LandmarkObs obs;
			obs = observations[j];
			x_obs = obs.x;
			y_obs = obs.y;
			ID_obs = obs.id;

			x_tObs = x_p + cos(theta_p) * x_obs - sin(theta_p) * y_obs;
			y_tObs = y_p + sin(theta_p) * x_obs + cos(theta_p) * y_obs;

			Transformed_Obs.push_back(LandmarkObs{ x_tObs, y_tObs, ID_obs });
		}
		// execute the data association to update the IDs for the observations.
		dataAssociation(predicted, Transformed_Obs);

		// Update the weight the selected particle.
		double alpha = 1.0;
		// Landmark stds
		double sigma_x, sigma_y;
		sigma_x = std_landmark[0];
		sigma_y = std_landmark[1];
		// Loop over all the Transformed_Obs to computer the likelihood of each observation, then assuming that
		// the observations are independent combine the likelihoods by mutiplying them to obtain the final weight.
		for (unsigned int j = 0; j < Transformed_Obs.size(); j++) {
			LandmarkObs obs;
			obs = Transformed_Obs[j];

			x_obs = obs.x;
			y_obs = obs.y;
			ID_obs = obs.id;

			// Loop over the predicted landmarks and computer the likelihood fo the observation for the correct ID.
			for (unsigned int k = 0; k < predicted.size(); k++) {
				
				if (ID_obs == predicted[k].id) {
					double x_pr;
					double y_pr;

					x_pr = predicted[k].x;
					y_pr = predicted[k].y;

					double likelihood;
					likelihood = (1.0 / (2 * M_PI * sigma_x * sigma_y)) *
								exp(-(pow(x_obs - x_pr ,2)+pow(y_obs - y_pr ,2)));
					// Assuming that the observations are independent combine them into a sinle likelihood function
					// by multiplying them.
					alpha *= likelihood;
				}
			}

		}
		// Update the weights
		weights[i] = alpha;
		p.weight = alpha;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
