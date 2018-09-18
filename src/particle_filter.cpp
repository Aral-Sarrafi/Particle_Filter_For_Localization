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

// Define a random engine to generate the required random numbers in the upcomming sections

static default_random_engine gen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Define the normal distribution for initialization noise
	normal_distribution<double> noise_x(0, std[0]);
	normal_distribution<double> noise_y(0, std[1]);
	normal_distribution<double> noise_theta(0, std[2]);

	// Loop over the number of particles and generate the particles.
	num_particles = 100;
	for (int i = 0; i < num_particles; i++) {
		Particle p;

		p.x = x + noise_x(gen);
		p.y = y + noise_y(gen);
		p.theta = theta + noise_theta(gen);
		p.id = i;
		p.weight = 1.0;


		particles.push_back(p);
		weights.push_back(1.0);

	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Define the noise distribution for the system model
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	// Loop over all the particles and predict the new particles based on the CTRV model.
	for (int i = 0; i < num_particles; i++) {
		if (fabs(yaw_rate) < 0.00001) {

			particles[i].x += velocity * cos(particles[i].theta) * delta_t;
			particles[i].y += velocity * sin(particles[i].theta) * delta_t;

		}
		else {

			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		// Add system noise to the predictions
		particles[i].x += noise_x(gen);
		particles[i].y += noise_y(gen);
		particles[i].theta += noise_theta(gen);

	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	// Loop over all the observations and compare each obeservation to all the predicted LandmarksObs to find the best
	// match for the selected obsevation, and update the ID for the observation.

	for (unsigned int i = 0; i < observations.size(); i++) {

		double x_obs = observations[i].x;
		double y_obs = observations[i].y;

		int ID = -1;

		double min_distance = numeric_limits<double>::infinity();

		for (unsigned int j = 0; j < predicted.size(); j++) {

			double x_pr = predicted[j].x;
			double y_pr = predicted[j].y;

			double distance = dist(x_obs, y_obs, x_pr, y_pr);

			if (distance < min_distance) {

				min_distance = distance;
				ID = predicted[j].id;
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

	// Loop over all the particles and update the weights.

	for (int i = 0; i < num_particles; i++) {

		// Get particle's state (x,y,theta)
		double x_p = particles[i].x;
		double y_p = particles[i].y;
		double theta_p = particles[i].theta;

		// For each particle loop over all the land marks and append the landmarks that are within the range of sensor.
		vector<LandmarkObs> predicted;

		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			double lm_x = map_landmarks.landmark_list[j].x_f;
			double lm_y = map_landmarks.landmark_list[j].y_f;
			int lm_ID = map_landmarks.landmark_list[j].id_i;

			double distance_to_particle = dist(x_p, y_p, lm_x, lm_y);

			if (distance_to_particle <= sensor_range) {
				predicted.push_back(LandmarkObs{ lm_ID, lm_x, lm_y });
			}
		}

		// Loop over all the observations and transform the observations to the particle coordinate system.
		vector<LandmarkObs> Transformed_obs;

		for (unsigned int j = 0; j < observations.size(); j++) {
			double tx = cos(theta_p)*observations[j].x - sin(theta_p)*observations[j].y + x_p;;
			double ty = sin(theta_p)*observations[j].x + cos(theta_p)*observations[j].y + y_p;;

			Transformed_obs.push_back(LandmarkObs{ observations[j].id, tx, ty });
		}
		// Update the ID for the transformed observations using the dataAssociation function.
		dataAssociation(predicted, Transformed_obs);

		// Loop over all the Transformed_obs and evaluate the likelihood of each observation.
		particles[i].weight = 1.0;
		weights[i] = 1.0;
		for (unsigned int j = 0; j < Transformed_obs.size(); j++) {

			double obs_x, obs_y, x_pr, y_pr;

			obs_x = Transformed_obs[j].x;
			obs_y = Transformed_obs[j].y;

			int correct_ID = Transformed_obs[j].id;

			// Loop over all the predicted landmarks and evaluate the likelihood for the correct IDs.


			for (unsigned int k = 0; k < predicted.size(); k++) {

				if (predicted[k].id == correct_ID) {

					x_pr = predicted[k].x;
					y_pr = predicted[k].y;

				}
			}
			double sigma_x = std_landmark[0];
			double sigma_y = std_landmark[1];

			double likelihood = (1.0 / (2 * M_PI * sigma_x * sigma_y)) *
				exp(-(pow(x_pr - obs_x, 2) / (2 * pow(sigma_x, 2)) + pow(y_pr - obs_y, 2) / (2 * pow(sigma_y, 2))));

			particles[i].weight *= likelihood;
			weights[i] *= likelihood;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Define the vector to store the new particles after resampling.
	vector<Particle> new_particles;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	// Resample the particles according to the weights.
	for (int i = 0; i < num_particles; i++) {

		int index = distribution(gen);
		new_particles.push_back(particles[index]);

	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
	const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1);  // get rid of the trailing space
	return s;
}
