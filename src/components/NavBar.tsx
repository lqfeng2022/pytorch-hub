import { Container, HStack, Text } from '@chakra-ui/react';
import { Link } from 'react-router-dom';

const NavBar = () => {
  return (
    <Container 
      maxW='1200px' 
      px='10px'
      position='fixed' 
      top='0' 
      left='0' 
      right='0' 
      zIndex='1000' 
      bg='white' // Ensure the background is white or your desired color
    >
      <HStack justifyContent='space-between' py='10px' my={2} spacing={3}>
        <Link to={'/'}>
          <Text 
            as='b' 
            whiteSpace='nowrap' 
            fontSize='2xl' 
            color='tomato'
            _hover={{ color: '#718096' }}
          >
            AI &nbsp;<Text as='span' fontWeight='thin'>WITH</Text> &nbsp;PYTORCH
          </Text>
        </Link>
        <Link to='https://simonlee88.netlify.app' target='_blank'>
          <Text as='b' fontSize='xl' color='gray.500' _hover={{ color: 'tomato' }}>SIMON
            <Text as='span' fontWeight='thin'> LEE</Text>
          </Text>
        </Link>
      </HStack>
    </Container>
  );
};

export default NavBar;
