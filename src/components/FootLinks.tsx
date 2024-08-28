import { HStack, Box, Text } from '@chakra-ui/react'
import { Link } from 'react-router-dom'

interface Props {
  left: string,
  leftLink: string,
  right: string,
  rightLink: string
}
const FootLinks = ({ left, leftLink, right, rightLink }: Props) => {
  const hoverStyle = { color: 'tomato', textDecoration: 'underline' }

  return (
    <HStack justifyContent='space-between' pb={5}>
      <Box>
        <Link to={leftLink}>
          <Text color='gray'>Previous Chapter</Text>
          <Text as='b' color='gray.600' _hover={hoverStyle}>
            {`<- ${left}`}
          </Text>
        </Link>
      </Box>
      <Box>
        <Link to={rightLink}>
          <Text textAlign='right' color='gray'>Next Chapter</Text>
          <Text as='b' color='gray.600' _hover={hoverStyle}>
            {`${right} ->`}
          </Text>
        </Link>
      </Box>
    </HStack>
  )
}

export default FootLinks