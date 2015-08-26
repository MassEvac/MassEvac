                def sim_shuffle_update(tstep):
                    '''Function to get update position of agents.'''

                    # Shuffle the order of the agents around
                    random.shuffle(self.S)
                    removal = []
                    # Record the density profile for this timestep in this list
                    this_tstep = [0]*fd.bins
                    for this_agent in self.S:
                        this_edge = self.E[this_agent]
                        if self.agent_update = 'shuffle':
                            this_density = self.density(this_edge)
                        elif self.agent_update
                        this_velocity = fd.v_dict[this_density]
                        
                        # Move the agent
                        this_location = self.L[this_agent] + this_velocity

                        di = self.h.D[this_edge]
                        # Travelling from this node (ri)
                        ri = self.h.R[this_edge]
                        # To this node (ci)
                        ci = self.h.C[this_edge]
                        # Has agent gone beyond the length of the link?
                        residual_di = this_location - di
                        # If the agent travels beyond the length of the link      
                        while residual_di > 0:
                            # Determine the amount of time remaining from last run
                            residual_time = residual_di/this_velocity
                    
                            # Agent has reached the end of the link
                            if ci == self.X[this_agent]:
                                this_location = di                                
                                
                                # Agent has reached its destination, record time
                                self.T[this_agent] = tstep + 1 - residual_time

                                # Subtract the agent from this edge
                                self.N[this_edge] -= 1
                        
                                # Remove agent from our list of agents
                                removal.append(this_agent)
                            # Agent has not reached the end of the link                                
                            else:
                            
                                next_ri = ci
                                next_ci = self.h.route[self.X[this_agent]][ci]
                                next_edge = self.h.EM[next_ri][next_ci]
                        
                                # If there is no space, then just wait at the end of the link
                                if self.N[next_edge]+1 > self.EC[next_edge]:
                                    this_location = di
                                # If there is enough space for the agent in the next link, only then proceed
                                else:             
                                    # Assign new nodes
                                    ri = next_ri
                                    ci = next_ci
                            
                                    # Subtract agent away from previous edge
                                    self.N[this_edge] -= 1
                                                        
                                    # Add agent to the new edge
                                    self.N[next_edge] += 1
                            
                                    # Update this edge
                                    this_edge = next_edge                            
                            
                                    # Update this density
                                    this_density = self.density(this_edge)
                            
                                    # If the next edge has vacancy then progress to the next edge
                                    # if this_density>kMax and random.random()<0.01: # 1% chance of rerouting
                                        # self.log_print('{0} {1} Congestion, rerouting...'.format(this_density, edge))
                                        # path=nx.single_source_dijkstra_path(JT,dest)
                            
                                    # Calculate new velocity
                                    this_velocity = fd.v_dict[this_density]
                            
                                    # Assign the new distance taking into account the remaining time
                                    this_location = this_velocity * residual_time
                            
                                    # Calculate new progress
                                    di = self.h.D[this_edge]

                            # Calculate the residual distance
                            residual_di = this_location - di

                        # Determine new position
                        this_node = np.array(self.h.nodes[ri])
                        that_node = np.array(self.h.nodes[ci])
                        
                        pr = this_location / di
                        offset = (that_node-this_node) * pr
                        self.P[this_agent,:] = this_node + offset

                        magnitude = np.linalg.norm(offset)
                        
                        if magnitude > 0:
                            unit = offset/magnitude
                        else:
                            unit = offset
                        
                        # Update array of unit vectors
                        self.O[this_agent,:] = unit
                
                        # Update to current agent location
                        self.L[this_agent] = this_location                
                
                        # Update to current agent edge
                        self.E[this_agent] = this_edge                

                        # Update to current agent edge
                        self.K[this_agent] = this_density
                
                        # Update journey time based on velocity on this link
                        if self.use_JT:
                            self.JT[ci][ri]['weight']=di/this_velocity
                
                        # Get the bin id in which to add the agent
                        bin = fd.which_bin(this_density)
                
                        # This is the density profile for a given agent over all timestep
                        self.KP_agent[this_agent,bin] = self.KP_agent[this_agent,bin] + 1
                
                        # This is the density profile for all agents per time step
                        this_tstep[bin] = this_tstep[bin] + 1
                
                        # This is the density profile for all agents for all edges over all timestep
                        self.KP_edges[this_edge,bin] = self.KP_edges[this_edge,bin] + 1
            
                    # Append the agglomerate density profile to our list of density profiles per timestep
                    self.KP_tstep.append(this_tstep)
            
                    # Remove agents that have reached destination
                    for r in removal:
                        self.S.remove(r)
        
